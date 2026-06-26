# Chapter 14 — Nonparametric & Flexible Models

### When you don't want to commit to a shape: splines, BART, and infinite mixtures — letting the data choose the functional form and the complexity

> *A note from me to you.*
>
> *Every model you have built so far began with a confession of belief. When you wrote `mu = a + b*x`, you were swearing an oath: "I believe the relationship between $x$ and $y$ is a straight line." When you added a quadratic, you upgraded the oath to "I believe it is a parabola." That oath is the most consequential decision in the whole analysis, and most of the time you have **no business making it**. You don't actually believe the number of bikes rented is a cubic polynomial in the hour of the day. You believe it goes up in the morning, peaks, dips at lunch, peaks again at the evening commute, and falls off at night — a shape with no name and no formula. The instant you force it into a polynomial, the polynomial fights back: it wiggles wildly at the edges, it invents oscillations the data never showed, and it extrapolates to the moon. You have all met this monster. Its name is Runge's phenomenon, and it is what happens when you make a promise about functional form that the data did not authorize.*
>
> *This chapter is about a different posture. Instead of committing to a shape and estimating a handful of parameters, we hand the question of shape itself over to inference. We build models whose **flexibility grows with the data** — give them more data and they will, if the data demand it, discover more structure; give them little and they stay humble. That is what "nonparametric" really means: not "no parameters" (there are often thousands), but "the number of effective parameters is not fixed in advance — it is learned." We will meet three workhorses that each answer a different flavor of "I refuse to guess the shape": **splines** (smooth curves built from local bumps, with a prior that controls how wiggly they're allowed to be), **BART** (a sum of hundreds of deliberately-weak decision trees that together carve out arbitrary nonlinear surfaces over many predictors), and the **Dirichlet process / infinite mixture** (the principled Bayesian answer to the question Chapter 13 left dangling: "how many clusters?" — answer: let the data tell you, with a model that has room for infinitely many).*
>
> *And because this is a course about judgment, not just code, I will end by sitting you down for the conversation no tutorial ever has: splines versus the Gaussian processes of Chapter 12 versus BART. They overlap. They compete. They each lie to you in a different way. By the end you will reach for the right one without thinking, the way a carpenter reaches past the hammer for the chisel.*

---

## What you'll be able to do after this chapter

- **Explain what "nonparametric" actually means** in the Bayesian sense — that model capacity grows with data, not that there are no parameters — and articulate *why* you'd let the data choose functional form rather than fixing it.
- **Build a Bayesian spline from scratch**: construct a cubic B-spline basis with `patsy.dmatrix`, place a coefficient prior on the basis weights, and read off a smooth flexible curve with honest uncertainty bands. You'll know exactly what a **knot** is, why we standardize, and the difference between an independent-Normal prior and a **random-walk (P-spline) smoothing prior** that *learns* how wiggly to be.
- **Fit a BART model** with `pymc_bart` — understand BART as a **sum of regularized weak trees**, run `pmb.BART` inside a `pm.Model`, let PyMC auto-assign the PGBART sampler, and produce **variable-importance** and **partial-dependence** plots to interpret an otherwise black-box fit.
- **Implement an infinite mixture via truncated stick-breaking** — the Dirichlet-process construction — and read the **concentration parameter** $\alpha$ as the dial that controls how many clusters the data are allowed to occupy.
- **Choose deliberately among splines, GPs (Chapter 12), and BART** using a decision table built on interpretability, smoothness assumptions, dimensionality, computational cost, and — the one everyone forgets — **extrapolation behavior**.

I assume you are fluent through **Chapter 13 (Mixture Models & Latent Variables)** — finite mixtures, the simplex weight vector, the `pm.Dirichlet` prior, label switching and the ordered-transform fix, and why PyMC marginalizes the discrete component label for you. I also assume you have met **Chapter 12 (Gaussian Processes)**: kernels, lengthscales, and the idea of a prior *over functions*. We lean on the standardization-by-default habit from **Chapter 03** and the diagnostic gates from **Chapter 07** ($\hat R \le 1.01$, ESS$_{\text{bulk}}$ and ESS$_{\text{tail}} \gtrsim 400$, zero divergences, $\hat k < 0.7$ for PSIS-LOO). One of the lessons of this chapter is *which of those gates relax* when you fit a model with thousands of nuisance parameters.

```python
import numpy as np
import pandas as pd
import pymc as pm
import pymc_bart as pmb
import arviz as az
import matplotlib.pyplot as plt
from patsy import dmatrix
import pytensor.tensor as pt

RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)
```

---

## 1. The motivation: parametric promises and the bill they come with

Let me make the problem visceral before I sell you the cure. We will fit a wiggly truth with a polynomial and watch it fail, because the failure is the entire argument for this chapter.

Suppose the true mean function is a smooth but un-nameable curve — say a couple of bumps — and we observe noisy samples from it:

```python
x = np.linspace(0, 1, 80)
f_true = np.sin(3 * np.pi * x) * 0.5 + 0.4 * np.exp(-((x - 0.7) ** 2) / 0.01)
y = f_true + rng.normal(0, 0.15, size=x.size)
```

A practitioner who insists on a parametric form must now *guess* the degree. Too low and they underfit (a line cannot bend twice). Too high and the polynomial overfits catastrophically — a degree-12 fit will thread every point and then explode beyond the data range. There is **no good degree** here, because the truth is not a polynomial at all. The honest statement is "I don't know the shape, and I don't want my prior to pretend I do."

> 💡 **Intuition:** A parametric model fixes a finite-dimensional shape and lets the data choose *where in that family* to sit. A nonparametric model fixes a *recipe for building shapes of growing complexity* and lets the data choose *how complex* to go. The polynomial says "pick a parabola." The spline says "I'll lay down a row of little local bumps; you tell me how tall each one is, and a prior will keep neighbors from disagreeing too violently." Same data, radically different humility.

Here is the deep idea that unifies the whole chapter, and it is worth slowing down for. **Three things in this course are all answers to the same question — "I don't want to commit to a parametric mean function, so give me a prior over flexible functions instead":**

| Approach | The prior over functions is... | Chapter |
|---|---|---|
| **Gaussian process** | a *direct* prior over functions, defined by a kernel that says how correlated nearby outputs are | 12 |
| **Spline / basis expansion** | a prior over the *coefficients* of a fixed set of local basis functions | 14 (here) |
| **BART** | a prior over *sums of decision trees*, each kept weak so the ensemble is smooth-ish but bumpy | 14 (here) |

A GP is the Platonic ideal: an infinite-dimensional prior over functions with exact uncertainty, but it costs $O(n^3)$ and groans in high dimensions. A spline is a *finite-dimensional approximation* to that same idea — lay down $K$ basis functions, and you have reduced "a prior over functions" to "a prior over $K$ numbers," which NUTS eats for breakfast. BART throws smoothness out the window in exchange for the ability to handle dozens of predictors with interactions and zero kernel-tuning. They are siblings, not rivals, and Section 7 is where we referee the family argument.

> 📜 **Citation/Origin:** The "let the data choose complexity" philosophy is the heart of Bayesian nonparametrics. For the spline view see McElreath, *Statistical Rethinking* 2e, Chapter 4 (the cherry-blossom example), and *Bayesian Modeling and Computation in Python* (Martin, Kumar, Lao), Chapter 5. For the GP↔spline↔BART connection, the cleanest modern treatment is BMCP Chapters 5 and 7. We will cite each in place.

A note on **what "nonparametric" does not mean.** It does not mean "no assumptions" — a spline assumes smoothness, a GP assumes the kernel, BART assumes additive tree structure. It does not mean "no parameters" — a 30-component stick-breaking mixture has well over a hundred. It means the *effective* number of parameters is governed by the data and a complexity prior, not nailed down a priori. Keep that straight and the rest is detail.

---

## 2. Bayesian splines: smooth curves from local bumps

The cleanest entry point to flexible modeling is the **basis expansion**, and the friendliest basis is the **B-spline**. The idea is almost embarrassingly simple once you see it, so let me build it from the ground up.

### 2.1 The core trick: replace one global curve with many local bumps

A polynomial is **global**: every coefficient affects the curve *everywhere*. Bump the cubic term and the fit at $x=0.1$ moves even if all your data disagreement was at $x=0.9$. That global coupling is the source of all the pathologies — the edge wiggles, the explosive extrapolation, the way one outlier in the corner drags the whole curve.

A spline is **local**. We chop the $x$-axis into segments at a set of breakpoints called **knots**, fit a low-degree polynomial (degree 3, cubic, by far the most common) *within each segment*, and glue the segments together so that the curve and its first two derivatives match at every knot — the joins are invisibly smooth. The genius of the **B-spline** ("basis spline") representation is that it accomplishes all of that gluing automatically by writing the curve as a weighted sum of pre-built **basis functions**:

$$
\mu(x) \;=\; a \;+\; \sum_{j=1}^{K} w_j \, B_j(x).
$$

Each $B_j(x)$ is a little hump that is **nonzero only over a few adjacent knot-segments and exactly zero everywhere else** (this is the "compact support" property — the thing polynomials lack). So the weight $w_j$ controls the height of the curve *only in $B_j$'s little neighborhood*. Locality, restored. In ASCII the model is just a linear regression on a clever design matrix:

```
mu(x) = a + B(x) @ w        # B is the (n_points x K) basis matrix; w are the K weights
```

> 💡 **Intuition:** Picture a row of tents pitched along the $x$-axis, each overlapping its neighbors. $B_j(x)$ is the height of tent $j$'s canvas above the point $x$; it's tall near tent $j$'s pole and zero once you walk past the adjacent poles. The fitted curve is the *sum of all the tent canvases*, each scaled by how high you've decided to pitch it ($w_j$). To make the curve go up at $x=0.7$, you raise the tents near $0.7$ — and crucially, you leave the tents near $0.1$ untouched. **That** is why splines don't get Runge's disease.

Two knobs define the basis, and you must understand both:

- **Degree.** Degree 3 (cubic) gives curves with continuous second derivatives — smooth to the eye and to the touch. Degree 1 gives a connect-the-dots polyline (continuous but kinked). Degree 0 gives a step function. Cubic is the default and almost always what you want for a smooth trend.
- **Knots: how many, and where.** The number of knots is the **complexity dial of the basis** — more knots → more (and narrower) tents → more local flexibility → more weights to estimate. The *placement* matters too: the standard, robust choice is to put knots at **quantiles of $x$**, so each tent covers roughly the same *number of data points* rather than the same width. That way dense regions get fine-grained tents and sparse regions get coarse ones — the resolution follows the data.

> ⚠️ **Pitfall:** Beginners think "I'll just crank the knots way up to be safe." Don't. With an independent-Normal prior on the weights, more knots means a wigglier, more overfit curve — you are buying back exactly the high-variance behavior you came here to escape. The disciplined fix is *either* keep the knots modest *or* (better) use the smoothing prior in §2.4 that lets the data damp the wiggles automatically regardless of knot count. We will do both and compare.

### 2.2 Building the basis with `patsy`

We do not hand-roll B-spline basis functions — the recurrence is fiddly and easy to get wrong. We use `patsy.dmatrix`, the same formula engine that powers `statsmodels` and (under the hood) much of Bambi. Its `bs(...)` helper builds a cubic B-spline design matrix given the data and the interior knots. Here is the canonical pattern, lifted from the PyMC splines how-to and adapted:

```python
num_knots = 15
# Knots at quantiles of x so each basis function covers ~equal data mass:
knot_list = np.quantile(x, np.linspace(0, 1, num_knots))

B = dmatrix(
    "bs(x, knots=knots, degree=3, include_intercept=True) - 1",
    {"x": x, "knots": knot_list[1:-1]},   # patsy wants the INTERIOR knots only
)
B = np.asarray(B, order="F")              # column-major -> fast matmul, see pitfall below
print(B.shape)                            # (80, K) -> one column per basis function
```

A few things in that call are load-bearing, and the dossier flags every one:

- **`knots=knot_list[1:-1]`** — `patsy`'s `bs()` wants the *interior* knots and adds the boundary knots itself from the data range. Passing the full quantile list (including the min and max) double-counts the boundaries and shifts your basis. Drop the first and last.
- **`include_intercept=True` ... `- 1`** — we ask `bs()` to include the constant basis function but then strip `patsy`'s *own* intercept column (`- 1`) so we don't get a redundant intercept. We will add the global intercept `a` ourselves in the model, which keeps the prior on it interpretable.
- **`np.asarray(B, order="F")`** — `patsy` returns a row-major (C-order) array; PyTensor's matmul has a faster path on column-major (Fortran-order) data. This is a pure performance nicety the PyMC gallery uses; it changes nothing about the math.

> 🩺 **Diagnostic:** Before you model anything, *plot the basis*. `plt.plot(x, B)` draws all $K$ tents. You should see a row of overlapping humps, each rising and falling, tiling the $x$-range, summing (because of `include_intercept`) to a constant. If you instead see columns that are flat zero, or humps piled up at one end, your knots are wrong — fix the basis before you waste a sampler run on it.

> ⚠️ **Pitfall:** Keep the basis matrix's column count and the model's `dims`/`size` **identical**. The number of B-spline columns is *not* simply `num_knots`; it's `num_interior_knots + degree + 1` (with the intercept convention above). Always read `B.shape[1]` off the actual matrix and feed *that* to the prior. Hardcoding the count is the fastest way to a silent shape bug — see the Common Errors table.

### 2.3 Prior choice #1 — independent Normal weights

Once we have $B$, the model is just a linear regression: $\mu = a + B\mathbf{w}$, with a likelihood on top. The *only* genuinely new modeling decision is the **prior on the weights $\mathbf w$**, and it is here that all the action lives — because the prior on $\mathbf w$ *is* the smoothness prior. It is the mathematical form of "how wiggly am I willing to let this curve be."

The simplest choice — and what the PyMC gallery uses — is an independent Normal on every weight:

```python
coords = {"splines": np.arange(B.shape[1])}
with pm.Model(coords=coords) as spline_indep:
    a = pm.Normal("a", mu=0.0, sigma=1.0)                      # global intercept
    w = pm.Normal("w", mu=0.0, sigma=1.0, dims="splines")      # one weight per basis fn
    mu = pm.Deterministic("mu", a + pm.math.dot(B, w))
    sigma = pm.Exponential("sigma", 1.0)
    pm.Normal("y", mu=mu, sigma=sigma, observed=y)
    idata_indep = pm.sample(1000, tune=1000, chains=4,
                            target_accept=0.9, random_seed=RANDOM_SEED)
```

This is honest and it works. But notice what it *assumes*: every weight is independent of its neighbors. Tent 5 has no idea what tent 6 is doing. Nothing in the prior says "adjacent tents should be at similar heights," which is the very definition of smoothness. So if you give this model many knots, neighboring weights are free to lurch up and down independently, and you get a wiggly fit — the prior's `sigma` is your only brake, and it's a blunt one (tighten it and you flatten *everything*, including real structure).

> 🔧 **In practice:** With the independent-Normal prior, smoothness is controlled by **two coupled knobs you must tune by hand**: the number of knots and the weight `sigma`. Standardize your predictor and outcome first (Chapter 03's habit) so that `sigma=1` on the weights is a sane, weakly-informative default rather than a wild guess. Then treat knot count as a model-comparison question (Chapter 11): fit a few, compare with `az.loo`, prefer the simplest that the data support.

> ⚠️ **Footgun — LOO needs the pointwise log-likelihood, and `pm.sample` does not store it by default.** Every time this chapter tells you to reach for `az.loo` / `az.compare` (here, and again in §3.3 and §5), remember that calling them on a vanilla `idata` raises `TypeError: ... log_likelihood not found in InferenceData`. You have two cures, both established back in Chapter 11: ask for it at sampling time —
> ```python
> idata = pm.sample(..., idata_kwargs={"log_likelihood": True})
> ```
> — or compute it after the fact on an existing trace:
> ```python
> pm.compute_log_likelihood(idata)   # adds idata.log_likelihood in place
> az.loo(idata)                      # now works
> ```
> This bites hardest for BART (§3.3), where the worked model already passes `compute_convergence_checks=False`; if you mean to compare it with LOO, you must opt the log-likelihood back in.

### 2.4 Prior choice #2 — the random-walk (P-spline) smoothing prior

Here is the better idea, and it is the one the topic brief and the dossier specifically ask for. Instead of treating the weights as independent, we tie each weight to its **neighbor** with a random walk:

$$
w_1 \sim \mathrm{Normal}(0, \sigma_w), \qquad w_j \;=\; w_{j-1} + \mathrm{Normal}(0,\, \tau), \quad j = 2,\dots,K,
$$

equivalently $w_j - w_{j-1} \sim \mathrm{Normal}(0,\tau)$. Read that out loud: *"the difference between adjacent weights is small, with a scale $\tau$ that I will learn from the data."* That is a **penalty on the first differences** of the weights — exactly the idea behind **P-splines** (penalized splines, Eilers & Marx 1996). A second-order random walk penalizes the *second* differences and prefers locally-linear trends; the first-order version above prefers locally-constant ones. Either way, the curve is now pulled toward smoothness *by the prior*, and the strength of that pull, $\tau$, is itself a parameter the data get to vote on.

> 🧮 **The math:** Why is "small adjacent differences" the same thing as "smooth"? Because the discrete second difference $w_{j+1} - 2w_j + w_{j-1}$ is a finite-difference approximation to the *second derivative* of the underlying curve, and a curve with small second derivative is, by definition, not very wiggly. Penalizing differences of the B-spline coefficients is a computationally trivial stand-in for penalizing $\int (\mu''(x))^2\,dx$ — the classical smoothing-spline roughness penalty. The random-walk prior is the Bayesian incarnation of that penalty, with the smoothing parameter $\tau$ promoted from a value-you-cross-validate to a value-you-infer. This is the spline's answer to the GP lengthscale.

In PyMC v5, write the random walk explicitly with `pm.GaussianRandomWalk`, and put a half-Cauchy or Exponential prior on the innovation scale $\tau$ so the data choose the smoothing strength:

```python
with pm.Model(coords=coords) as spline_smooth:
    a = pm.Normal("a", mu=0.0, sigma=1.0)
    tau = pm.HalfCauchy("tau", beta=1.0)                       # learned smoothing strength
    # Adjacent weights tied: w_j = w_{j-1} + Normal(0, tau).
    w = pm.GaussianRandomWalk("w", sigma=tau, init_dist=pm.Normal.dist(0, 1),
                              dims="splines")
    mu = pm.Deterministic("mu", a + pm.math.dot(B, w))
    sigma = pm.Exponential("sigma", 1.0)
    pm.Normal("y", mu=mu, sigma=sigma, observed=y)
    idata_smooth = pm.sample(1000, tune=1000, chains=4,
                             target_accept=0.9, random_seed=RANDOM_SEED)
```

Now you can be generous with knots — even reckless — and the curve stays smooth, because $\tau$ shrinks toward small values unless the data *insist* on rapid local change. Small $\tau$ ⇒ adjacent weights nearly equal ⇒ a nearly-flat (very smooth) curve; large $\tau$ ⇒ weights free to roam ⇒ a wiggly curve. The data move $\tau$ to wherever the bias-variance tradeoff is best, and you get a posterior over that tradeoff instead of a single cross-validated point estimate. This is the spline doing what GPs and BART do: **learning its own complexity.**

> ⚠️ **Pitfall:** A random-walk prior on the weights can mildly correlate with the global intercept `a` and the residual `sigma`, and the innovation scale `tau` lives near a soft funnel (Chapter 07) — when `tau` is tiny the walk collapses and the geometry pinches. If you see a handful of divergences, raise `target_accept` to 0.95–0.99 first, and consider a **non-centered parameterization** of the walk (sample standardized innovations and scale by `tau`) exactly as you learned for hierarchical models in Chapter 10. The funnel never really leaves you; it just changes costume.

Here is that non-centered walk written out by hand. The trick is identical to the hierarchical non-centering of Chapter 10: instead of sampling the correlated weights directly, sample **standardized innovations** `z_j ~ Normal(0, 1)` whose geometry doesn't depend on `tau`, then build the walk as a scaled cumulative sum so `tau` enters multiplicatively rather than as the scale of the thing NUTS is exploring:

```python
with pm.Model(coords=coords) as spline_smooth_nc:
    a = pm.Normal("a", mu=0.0, sigma=1.0)
    tau = pm.HalfCauchy("tau", beta=1.0)
    # Standardized innovations — unit-scale, funnel-free geometry:
    z = pm.Normal("z", mu=0.0, sigma=1.0, dims="splines")
    # Reconstruct the random walk: w_j = tau * sum_{i<=j} z_i  (cumulative sum).
    w = pm.Deterministic("w", tau * pt.cumsum(z), dims="splines")
    mu = pm.Deterministic("mu", a + pm.math.dot(B, w))
    sigma = pm.Exponential("sigma", 1.0)
    pm.Normal("y", mu=mu, sigma=sigma, observed=y)
    idata_smooth_nc = pm.sample(1000, tune=1000, chains=4,
                                target_accept=0.9, random_seed=RANDOM_SEED)
```

This is *mathematically the same prior* as the `GaussianRandomWalk` above — first differences `w_j - w_{j-1} = tau * z_j ~ Normal(0, tau)` — but NUTS now explores the well-conditioned `z`-space and multiplies by `tau` afterward, so the funnel that pinched the centered walk at small `tau` largely dissolves. When a centered spline walk keeps throwing divergences even at `target_accept=0.99`, this is the cure, and it is exactly the move you already know from the Eight Schools funnel.

### 2.5 Worked example: cherry-blossom bloom dates

Let me put it all together on a named dataset with real history behind it. The **cherry-blossom** series records the day-of-year (`doy`) on which Kyoto's cherry trees first bloomed, going back over a millennium — one of the longest phenological records in existence, and a McElreath favorite (Rethinking 2e, Ch. 4). We want the *smooth trend in bloom date over the centuries*. We have no theory for the shape (it reflects slow climate drift, with the recent warming pulling blooms earlier), so a spline is exactly right.

```python
blossom = pd.read_csv(pm.get_data("cherry_blossoms.csv"))   # verify the bundled CSV name
blossom = blossom.dropna(subset=["doy"]).reset_index(drop=True)
year = blossom["year"].to_numpy().astype(float)
doy = blossom["doy"].to_numpy().astype(float)
print(blossom[["year", "doy"]].describe())
```

`doy` averages around 105 (early-to-mid April) with a standard deviation near 6 days. Now the basis, with knots at year-quantiles:

```python
num_knots = 15
knot_list = np.quantile(year, np.linspace(0, 1, num_knots))
B = dmatrix(
    "bs(year, knots=knots, degree=3, include_intercept=True) - 1",
    {"year": year, "knots": knot_list[1:-1]},
)
B = np.asarray(B, order="F")
coords = {"splines": np.arange(B.shape[1]), "obs": np.arange(year.size)}
```

Because `doy` is on a natural scale around 105, we center the intercept prior there instead of standardizing the outcome — a small, defensible departure from the standardize-everything rule that keeps the numbers human-readable. Predictor knots are at quantiles so scale is handled.

```python
with pm.Model(coords=coords) as blossom_spline:
    a = pm.Normal("a", mu=105.0, sigma=10.0)          # intercept near the mean bloom day
    tau = pm.HalfCauchy("tau", beta=1.0)              # smoothing strength, learned
    # init_dist sets the prior on the FIRST weight w_1, hence the curve's overall
    # offset from the intercept a. We loosen its scale to 5 (vs 1 in §2.4) because
    # doy is on its natural ~6-day-SD scale here, not the unit scale of §2.4.
    w = pm.GaussianRandomWalk("w", sigma=tau, init_dist=pm.Normal.dist(0, 5),
                              dims="splines")
    mu = pm.Deterministic("mu", a + pm.math.dot(B, w), dims="obs")
    sigma = pm.Exponential("sigma", 1.0)
    pm.Normal("D", mu=mu, sigma=sigma, observed=doy, dims="obs")
```

**Always do the prior predictive check first** (Chapter 03, Chapter 08). It catches absurd priors before they cost you a sampler run:

```python
with blossom_spline:
    prior = pm.sample_prior_predictive(draws=200, random_seed=RANDOM_SEED)

# Overplot 50 prior-implied mu curves over the years. Stacking (chain, draw) into a
# single 'sample' axis lets us index draws regardless of the chain/draw split:
mu_prior = prior.prior["mu"].stack(sample=("chain", "draw")).values   # (obs, samples)
for i in range(50):
    plt.plot(year, mu_prior[:, i], color="C0", alpha=0.1)
plt.xlabel("year"); plt.ylabel("prior mu (first-bloom day of year)")
```

> 🩺 **Diagnostic — reading the prior predictive:** Overplot ~50 of the prior `mu` curves against `year`. You want curves that are **smooth, of plausible amplitude (bloom dates should wander within a couple of weeks, not swing 200 days), and centered near 105**. If your prior curves are flat lines, `tau`'s prior is too tight — the data will struggle to add wiggle. If they're seismograph noise, `tau` is too loose — tighten the HalfCauchy `beta`. The random-walk prior should produce gently meandering curves: that *is* the visual signature of "I believe in a smooth trend of unknown shape."

Now sample and gate on diagnostics:

```python
with blossom_spline:
    idata = pm.sample(1000, tune=1000, chains=4,
                      target_accept=0.95, random_seed=RANDOM_SEED)
    idata.extend(pm.sample_posterior_predictive(idata, random_seed=RANDOM_SEED))

az.summary(idata, var_names=["a", "tau", "sigma"])
```

> 🩺 **Diagnostic — what to demand here.** Check `a`, `tau`, `sigma` (the *interpretable* parameters) for $\hat R \le 1.01$ and ESS $\gtrsim 400$. Concretely, you should see `a` land near 105 (the mean bloom day you read off `describe()`), `sigma` settle around 5–6 days (the residual scatter the smooth trend doesn't explain), and a **small `tau`** — the data want a gently smooth trend, not a seismograph, so the learned innovation scale comes out modest. All three should report $\hat R = 1.00$ and ESS in the thousands. The individual spline weights `w` are nuisance parameters whose marginals you do not interpret — modest ESS on them is fine *as long as* the curve `mu` is well-behaved, because `mu` is what you actually report. This is your first taste of a recurring theme: **in flexible models, gate the quantities you interpret, not every internal coordinate.** A couple of divergences from `tau`'s soft funnel are tolerable; a wall of them means reparameterize (the non-centered walk above).

Finally, the payoff — the posterior trend with uncertainty:

```python
mu_post = idata.posterior["mu"]                       # (chain, draw, obs)
mu_mean = mu_post.mean(("chain", "draw")).values
hdi = az.hdi(idata, var_names=["mu"])["mu"]            # 94% HDI band per year

plt.scatter(year, doy, s=8, alpha=0.3)
plt.plot(year, mu_mean, color="C1", lw=2)
plt.fill_between(year, hdi.sel(hdi="lower"), hdi.sel(hdi="higher"),
                 color="C1", alpha=0.3)
plt.xlabel("year"); plt.ylabel("first-bloom day of year")
```

> 🩺 **Diagnostic — reading the fitted curve.** The orange line is the posterior-mean trend; the band is the 94% HDI of the *mean function* `mu` (not of new observations — for that, use the posterior predictive `D`). You should see a slowly undulating curve that dips noticeably in the most recent centuries — blooms arriving earlier, the fingerprint of warming. Crucially, look at the **band width**: it is narrow where the data are dense and *fans out where data are sparse* (the early years with few records). That widening is the model being honest about ignorance — a parametric polynomial would report false confidence there. Sit with that for a second: the spline does not just fit a curve, it tells you *where it doesn't know*. That is the whole reason we went Bayesian.

> ⚠️ **Pitfall — extrapolation.** Push `year` beyond the data range and the spline does something dangerous and dumb: outside the boundary knots the basis functions go flat or run off linearly, so your "forecast" is an artifact of the basis, not a belief about the future. **Splines interpolate beautifully and extrapolate terribly.** If you need principled extrapolation with growing uncertainty, that is a job for a GP (Chapter 12) or a structural time-series model (Chapter 15) — file this under §7's decision table.

---

## 3. BART: a sum of deliberately-weak trees

Splines and GPs are wonderful in one or two dimensions and start to creak past three or four — you cannot place knots in a 30-dimensional space, and a GP's kernel becomes a nightmare to specify. The moment your problem looks like a **tabular dataset with many predictors, nonlinear effects, and interactions you didn't anticipate**, you want a different kind of flexible model. You want **BART**: Bayesian Additive Regression Trees.

### 3.1 What BART is, and why "weak" is the whole point

You know decision trees from ML: a tree recursively splits the predictor space ("is `temperature > 20`? then is `hour > 17`?") and predicts a constant in each leaf. A single deep tree is a high-variance overfitting machine. The boosting world (gradient boosting, XGBoost) tames this by summing **many shallow trees**, each correcting the last. BART is the **Bayesian** member of that family, and its model is exactly:

$$
f(\mathbf x) \;\approx\; \sum_{j=1}^{m} g_j(\mathbf x;\, T_j, M_j),
$$

a **sum of $m$ regression trees**, where $T_j$ is the structure (splits) of tree $j$ and $M_j$ are its leaf values. We then wrap $f(\mathbf x)$ in whatever likelihood the data need — Normal for regression, Poisson/NegBin for counts, Bernoulli for classification.

The single most important idea — the thing that separates BART from a pile of overfit trees — is the **regularization prior that keeps each tree deliberately weak**. BART places a prior on tree *depth* that makes deep trees very unlikely:

$$
P(\text{a node at depth } d \text{ splits}) \;=\; \alpha\,(1 + d)^{-\beta},
$$

with defaults $\alpha = 0.95$ and $\beta = 2$. Read it: the root (depth 0) splits with probability $0.95$, a depth-1 node with probability $0.95 \cdot 2^{-2} \approx 0.24$, a depth-2 node with probability $0.95 \cdot 3^{-2} \approx 0.11$, and so on — the prior strangles depth, so each individual tree is a **stump-like weak learner that explains only a sliver of the signal**. No single tree can dominate; the *sum* of hundreds of these humble trees is what flexibly fits the surface. There is also a shrinkage prior on the leaf values so each tree contributes only a small increment.

> 💡 **Intuition:** Think of BART as a committee of $m$ cautious advisors. Each advisor (tree) is forbidden by the prior from being too confident or too detailed — they can only offer a crude, shallow opinion. But average a few hundred crude-but-diverse opinions and you recover a remarkably nuanced, nonlinear consensus, *with* the disagreement among posterior draws giving you honest uncertainty bands for free. Weakness is a feature: it is what makes the ensemble generalize instead of memorize. This is the same wisdom as random forests and boosting, dressed in a prior and delivering a full posterior.

> 📜 **Citation/Origin:** BART is due to Chipman, George & McCulloch (2010). The PyMC implementation, `pymc-bart`, and its PGBART sampler are described in Quiroga, Garay, Alonso, Loyola & Martin, *"Bayesian additive regression trees for probabilistic programming"* (arXiv:2206.03619). The PGBART (Particle Gibbs) sampler is what lets BART live inside a general PPL alongside ordinary continuous parameters — a genuinely modern piece of engineering.

### 3.2 The `pymc_bart` API

`pymc-bart` exposes BART as a single distribution-like object you drop into a `pm.Model`. The headline facts (all verified against the current API in the dossier):

- **`pmb.BART(name, X, Y, m=50, alpha=0.95, beta=2.0, response='constant', split_rules=None, split_prior=None)`** creates the BART random variable. `X` is the $(n \times p)$ covariate matrix, `Y` the response, `m` the number of trees.
- **`m` is your main dial.** More trees → more expressive and lower-variance (and slower); the fit stays fundamentally **piecewise-constant** regardless of `m`, but with more trees the individual step artifacts average down so it *looks* a little smoother. Don't read "more trees" as "becomes a smooth curve" — BART is step-like by construction (see the §5 "not smooth" row and §3.4). 50 is a sane default; bump to 100–200 for harder surfaces.
- **BART lives on the link scale.** The output `μ` is a latent function. For Gaussian regression you can use it directly as the mean. For counts you model `np.log(Y)` and wrap with `pm.math.exp(μ)`; for classification you wrap with `pm.math.invlogit(μ)`. **Forgetting the link wrap is the #1 BART bug** — you'll get negative "rates" feeding a Poisson and a sampler error.
- **You do not choose a sampler.** `pm.sample()` auto-detects the BART RV and assigns the **PGBART** step to it, while NUTS handles the other continuous parameters (like a NegBin's dispersion `α`). It just works.
- **`compute_convergence_checks=False`** is commonly passed: the thousands of per-tree parameters are *not* your inferential target, so the usual $\hat R$/ESS warnings on them are noise. You judge a BART fit by **posterior predictive checks and held-out predictive accuracy**, not by $\hat R$ on tree internals.

> ⚠️ **API drift warning (per the dossier).** `pymc-bart` is an actively developed package and its helper-function signatures *do* move between releases — the exact spellings of `compute_variable_importance(... method=...)`, the `response` options, and the `plot_*` keyword arguments have changed and may change again. Treat the snippets below as the canonical current shape and **`# verify in current pymc-bart docs`** before you rely on a specific keyword. The *model* call `pmb.BART("μ", X, Y, m=...)` is stable; the analysis helpers are the moving target.

### 3.3 Worked example: bike rentals with BART

The **bikes** dataset (bundled with PyMC; the `pymc-bart` introduction uses it) records bicycle-rental `count` against a handful of predictors — `hour` of day, `temperature`, `humidity`, and `weekday`. Rentals are **counts** (so NegBin, to absorb overdispersion — recall Chapter 09), and the relationship to `hour` is wildly nonlinear (twin commute peaks) with `temperature` interacting (nobody rides in the cold). This is BART's home turf: many features, nonlinear, interacting, and we have no parametric shape in mind.

```python
bikes = pd.read_csv(pm.get_data("bikes.csv"))          # verify bundled CSV name/columns
print(bikes.columns.tolist())                          # CONFIRM the column names before indexing
features = ["hour", "temperature", "humidity", "weekday"]
X = bikes[features].to_numpy()
Y = bikes["count"].to_numpy().astype(float)
```

> ⚠️ **Dataset provenance — confirm the columns.** The `bikes` data ships in slightly different forms across the ecosystem: the Bambi example and the `pymc-bart` example do **not** use identical column sets. The four feature names above match the `pymc-bart` introduction; if `print(bikes.columns)` shows different names (e.g. `temp` instead of `temperature`, `hr` instead of `hour`, or `cnt` instead of `count`), adapt `features` and the `Y` column accordingly — otherwise every downstream line raises a `KeyError`. Always look before you index.

The model — note the **`np.log(Y)`** passed to BART and the **`pm.math.exp`** wrap, so the latent BART function lives on the log-rate scale and the rate fed to the NegBin is guaranteed positive:

```python
with pm.Model() as model_bikes:
    alpha = pm.Exponential("alpha", 1.0)                       # NegBin dispersion
    mu = pmb.BART("mu", X, np.log(Y), m=50)                    # latent log-rate surface
    y = pm.NegativeBinomial("y", mu=pm.math.exp(mu), alpha=alpha, observed=Y)
    idata_bart = pm.sample(draws=2000, tune=500,
                           compute_convergence_checks=False,
                           random_seed=RANDOM_SEED)
    idata_bart.extend(pm.sample_posterior_predictive(idata_bart,
                                                     random_seed=RANDOM_SEED))
```

That is the *entire* model. No knots to place, no kernel to choose, no interaction terms to specify — BART discovers the twin-peak hour effect and its temperature interaction on its own. That minimal-tuning property is BART's superpower.

**Judging the fit.** Because we suppressed convergence checks, we lean on the **posterior predictive check**:

```python
az.plot_ppc(idata_bart, num_pp_samples=100)
```

> 🩺 **Diagnostic — reading the BART PPC.** `az.plot_ppc` overlays the distribution of replicated counts (thin lines) on the observed count distribution (thick line). For a good BART count fit the replicated and observed densities should sit on top of each other across the range, including the right tail of high-rental hours. If the replicates systematically undershoot the peak counts, you may need more trees (`m`) or you've mis-specified the likelihood (try the dispersion). If they over-smooth a sharp feature, again more trees. BART rarely *underfits* with `m=50`+; when it disappoints it's usually a likelihood-scale issue, not a flexibility issue.

#### Interpreting BART: variable importance

A single sum-of-200-trees function is a black box. The first interpretive tool is **variable importance** — which predictors actually drove the fit:

```python
vi_results = pmb.compute_variable_importance(idata_bart, mu, X)   # # verify in current pymc-bart docs
pmb.plot_variable_importance(vi_results)
```

> 🩺 **Diagnostic — reading variable importance.** The plot ranks predictors by how much predictive accuracy ($R^2$) you *lose* when that variable is dropped from the trees (the backward-selection variant) or by how often it's chosen for splits (the counting heuristic). For bikes you should see `hour` dominate, `temperature` second, `humidity` and `weekday` trailing. Two cautions: (1) importance is **relative and model-internal**, not a causal claim — a variable can be "important" because it proxies for an omitted cause; (2) correlated predictors **split their importance**, so a low score doesn't prove irrelevance. Use it to prune and prioritize, not to adjudicate science.

#### Interpreting BART: partial dependence

Variable importance says *which* predictors matter; **partial-dependence plots (PDPs)** say *how*. A PDP marginalizes over all the other predictors and shows the average effect of one (or two) on the response:

```python
pmb.plot_pdp(mu, X=X, Y=Y, grid=(2, 2), func=np.exp,       # exp -> back to count scale
             var_discrete=[3])                              # 'weekday' is discrete (index 3)
```

> 🩺 **Diagnostic — reading the PDP.** Each panel is one predictor; the curve is the average predicted rental count as that predictor sweeps its range, *holding the data distribution of the others fixed*. The `hour` panel should show the unmistakable **double hump** — a morning commute peak and a larger evening one — that no polynomial would have found without you hand-coding it. `temperature` should rise then plateau or dip at extreme heat. The shaded band is posterior uncertainty: wide where data are sparse. Two warnings the PDP literature insists on: PDPs **average over interactions**, so a flat PDP can hide a strong effect that only appears in combination with another variable (use a 2-D PDP or `plot_ice` to expose it); and PDPs are **unreliable in regions where predictors are correlated** because the marginalization synthesizes feature combinations the data never contained. For honest *per-observation* curves rather than the average, reach for `pmb.plot_ice` (individual conditional expectation), which draws one line per observation and reveals heterogeneity the PDP's averaging erases.

> 🔧 **In practice:** The workflow for a BART model is: (1) fit with a sensible `m`; (2) PPC to confirm it fits; (3) variable importance to find the drivers and optionally drop dead weight; (4) PDP/ICE on the top drivers to *understand* the shapes; (5) report held-out predictive performance (`az.loo`, but watch $\hat k$ — Chapter 11) for comparison against rivals. **Note the trap:** the model above passes `compute_convergence_checks=False`, and `pm.sample` never stored the pointwise log-likelihood — so before `az.loo`/`az.compare` will run you must add it, exactly as flagged in §2.3:
> ```python
> pm.compute_log_likelihood(idata_bart)   # or pass idata_kwargs={"log_likelihood": True} to pm.sample
> az.loo(idata_bart)
> ```
> You get the predictive punch of gradient boosting with a *posterior* attached, which is the entire reason to pay BART's compute cost over XGBoost.

### 3.4 Splines vs BART, head to head, on the same 1-D problem

To make the contrast concrete, fit *both* a spline and a BART model to the same single-predictor nonlinear data from §1 (`x`, `y`), then overlay their fits:

```python
# Spline fit on (x, y): reuse the smoothing-prior model from §2.4 -> idata_smooth, mu
# BART fit on the same data (1-D X), Gaussian likelihood:
X1 = x.reshape(-1, 1)
with pm.Model() as model_1d_bart:
    sig = pm.HalfNormal("sig", 1.0)
    f = pmb.BART("f", X1, y, m=50)
    pm.Normal("y_obs", mu=f, sigma=sig, observed=y)
    idata_1d_bart = pm.sample(draws=2000, tune=500,
                              compute_convergence_checks=False,
                              random_seed=RANDOM_SEED)
```

> 🩺 **Diagnostic — what the overlay teaches.** Plot the truth `f_true`, the spline posterior-mean curve, and the BART posterior-mean curve together. You will see the defining stylistic difference: the **spline is smooth** — it glides through the bumps because smoothness is baked into its prior — while **BART is bumpy and locally flat with little step-like jumps**, because it is, underneath, a sum of piecewise-constant trees. On smooth 1-D signals the spline usually wins on visual fidelity and gives tighter, more honest bands. Now imagine the same picture with *twenty* predictors and unknown interactions: the spline can't even be built, and BART sails through unchanged. **That trade — smoothness-in-low-dimensions versus flexibility-in-high-dimensions — is the entire spline-vs-BART decision**, and §7 formalizes it.

---

## 4. Infinite mixtures: the Dirichlet process and "how many clusters?"

Chapter 13 left you with an honest discomfort. You fit a finite mixture, but you had to *pick $K$*, the number of components, in advance — and picking $K$ is exactly the kind of "promise about complexity the data didn't authorize" this chapter exists to abolish. You could fit several $K$ and compare with LOO, but that is a workaround. The properly Bayesian answer is a model with **room for infinitely many components**, where the data decide how many are actually *occupied*. That model is the **Dirichlet process (DP) mixture**, and the trick that makes it computable is **stick-breaking**.

### 4.1 The stick-breaking construction

Imagine a stick of length 1 — your total probability mass. You break off a fraction $\beta_1$ for component 1, then break off a fraction $\beta_2$ of *what remains* for component 2, and so on forever. The weights are:

$$
w_1 = \beta_1, \qquad w_k = \beta_k \prod_{j<k}(1 - \beta_j), \qquad \beta_k \sim \mathrm{Beta}(1, \alpha).
$$

Each $\beta_k$ is the fraction of the *remaining* stick that component $k$ claims. Because the leftover stick shrinks geometrically, **later components get exponentially less mass** — most of the probability piles onto the first few components, and the tail components are effectively empty. That is the magic: the construction *automatically* favors using few components, but never forbids using more if the data push back. This is sometimes called the **GEM distribution** (Griffiths–Engen–McCloskey).

> 🧮 **The math:** Why does this self-truncate? After $k$ breaks, the expected remaining stick length is $\mathbb{E}\!\left[\prod_{j\le k}(1-\beta_j)\right] = \left(\tfrac{\alpha}{1+\alpha}\right)^{k}$, which decays geometrically. So component weights shrink geometrically in expectation, and the number of components with non-negligible mass is finite with probability 1 — even though the model *defines* infinitely many. The expected number of occupied components in $N$ data points grows only like $\alpha \log N$. You get an unbounded model whose *effective* size is data-driven and modest: the textbook definition of Bayesian nonparametrics.

### 4.2 The concentration parameter $\alpha$ — the one dial that matters

Everything about a DP mixture's behavior is governed by the **concentration parameter** $\alpha$ in $\mathrm{Beta}(1, \alpha)$:

- **Small $\alpha$** (e.g. 0.5): each $\beta_k$ tends to be *large* (Beta(1, small) skews toward 1), so the first stick-break grabs most of the mass → **few, dominant clusters**.
- **Large $\alpha$** (e.g. 20): each $\beta_k$ tends to be *small*, so mass spreads across **many clusters**.

Because we rarely know the right granularity in advance, we put a prior on $\alpha$ itself — typically $\alpha \sim \mathrm{Gamma}(1, 1)$ — and let the data infer how many clusters to spend. *That* is the model answering "how many clusters?" for you, with a posterior over the answer rather than a single guess.

> 💡 **Intuition:** $\alpha$ is a "newness tax." A new data point either joins an existing cluster (with probability proportional to that cluster's current size — the rich get richer) or starts a brand-new cluster (with probability proportional to $\alpha$). Small $\alpha$ makes starting new clusters expensive, so you get a few big ones; large $\alpha$ makes novelty cheap, so clusters proliferate. This "Chinese restaurant process" metaphor — customers sitting at tables, popular tables attracting more customers, $\alpha$ governing how often someone starts a new table — is the same DP viewed from the data's perspective.

### 4.3 Truncated stick-breaking in PyMC

We cannot represent literally infinite components on a computer, so we **truncate** at a level $K$ large enough that the leftover sticks carry negligible weight — and then *verify* that they do. Here is the verified PyMC v5 pattern (note `pytensor.tensor as pt`, **not** the long-dead `theano`/`aesara`):

```python
def stick_breaking(beta):
    # w_k = beta_k * prod_{j<k}(1 - beta_j)
    portion_remaining = pt.concatenate([[1], pt.extra_ops.cumprod(1 - beta)[:-1]])
    return beta * portion_remaining

K = 30                                   # truncation level (an upper bound on # clusters)
N = x.size
coords = {"component": np.arange(K), "obs_id": np.arange(N)}

with pm.Model(coords=coords) as dp_mixture:
    alpha = pm.Gamma("alpha", 1.0, 1.0)                      # DP concentration
    beta = pm.Beta("beta", 1.0, alpha, dims="component")     # stick fractions
    w = pm.Deterministic("w", stick_breaking(beta), dims="component")

    # Component parameters (Normal components, Normal-Gamma style hyperprior):
    tau = pm.Gamma("tau", 1.0, 1.0, dims="component")
    lambda_ = pm.Gamma("lambda_", 10.0, 1.0, dims="component")
    mu = pm.Normal("mu", 0.0, sigma=10.0, dims="component")

    obs = pm.NormalMixture("obs", w=w, mu=mu, sigma=1.0 / pt.sqrt(lambda_ * tau),
                           observed=x_std, dims="obs_id")     # x_std = standardized data
    idata_dp = pm.sample(1000, tune=2000, target_accept=0.95,
                         random_seed=RANDOM_SEED)
```

`pt.extra_ops.cumprod` computes the running product of the leftover-stick fractions; `pt.concatenate([[1], ...[:-1]])` prepends a 1 so component 1 gets the *whole* stick before any is broken off. `pm.NormalMixture` then does the same automatic marginalization of the discrete cluster label you learned in Chapter 13 — there is no explicit `z`.

> 🩺 **Diagnostic — did I truncate high enough?** Run `az.plot_forest(idata_dp, var_names=["w"])`. You want to see the **last several components pinned near zero weight** — that is your proof the truncation $K$ was generous enough that the infinite tail you chopped off was empty anyway. If the highest-index sticks still carry real mass, *raise $K$* and refit; you truncated too aggressively and may be missing clusters.

> ⚠️ **Pitfall — DP mixtures are gloriously badly-behaved samplers.** Expect divergences, low ESS, weak $\hat R$ on the component-level parameters, and multimodality. This is *not* (only) your bug — it is the same label-switching and funnel pathology from Chapter 13, now with $K$ up to 30 overlapping, partly-empty components and a concentration funnel on top. The saving grace, and the reason the field tolerates it for **density estimation**: the *density* $\sum_k w_k\, f(\cdot\mid\theta_k)$ is identified and well-estimated even when the individual labels are hopeless. So judge a DP mixture by its **estimated density** (overlay the posterior-predictive density on a histogram of the data) and by the **forest plot of weights**, not by $\hat R$ on `mu`. Raise `target_accept` to 0.95–0.99, lengthen `tune`, standardize the data, and make peace with messy chains. If you genuinely need clean cluster labels (not just a density), the ordered transform from Chapter 13 or post-hoc relabeling is your recourse — but for "what does this distribution look like and roughly how many modes does it have," the DP mixture is the right, principled tool.

---

## 5. The decision guide: splines vs GP vs BART

This is the section you will come back to. All three give you "a prior over flexible functions," and the right choice depends on five axes. Here is the table I actually use, followed by the reasoning.

| Axis | **Splines** | **Gaussian Process (Ch 12)** | **BART** |
|---|---|---|---|
| **Smoothness** | Smooth by construction; smoothness *learned* via the RW/P-spline prior $\tau$ | Smoothness *assumed* via the kernel (e.g. ExpQuad ⇒ infinitely differentiable); the cleanest, most principled smoothness | Bumpy / piecewise-constant; **not** smooth — a sum of step functions |
| **Dimensionality** | Great in 1–2D; awful past ~3D (can't place knots in a high-D grid) | Excellent in low-D; degrades and gets expensive in high-D | **Shines in high-D / many features**; handles interactions automatically |
| **Interpretability** | High — a curve you can read; coefficients are local | Medium — lengthscale/amplitude are interpretable, the function less so | Low per-se, but rescued by **variable importance + PDP/ICE** |
| **Cost** | Cheap — finite linear model, NUTS-friendly | Expensive — $O(n^3)$ exact; use **HSGP / sparse** (Ch 12) to scale | Moderate — PGBART is not free but scales to many features |
| **Extrapolation** | **Terrible** — basis goes flat/linear past the knots | **Good & honest** — reverts to the mean with *growing* uncertainty | **Poor** — predicts the nearest leaf's constant; flat, overconfident outside the data |
| **Uncertainty quality** | Good interpolation bands; false confidence in extrapolation | **Best** — exact, principled, widens where data are sparse | Good ensemble bands; can be overconfident at edges |
| **Tuning burden** | Knots + prior; modest | Kernel choice + lengthscale prior; can be fiddly | **Lowest** — `m` and go |

> 💡 **Intuition — the one-line rules of thumb.**
> - **Smooth trend in 1–2 dimensions, want interpretability and speed?** → **Spline.** (Bloom dates over years; a dose-response curve; a seasonal shape.)
> - **Smooth function in low-D where you need the *best, most honest* uncertainty and principled extrapolation?** → **GP** (Chapter 12). (CO₂ forecasting; spatial interpolation; anywhere "how sure am I out here?" is the question.)
> - **Tabular data, many predictors, unknown nonlinearities and interactions, minimal tuning, predictive accuracy is king?** → **BART.** (The bikes problem; most "I have 25 columns and want a good Bayesian predictor" situations.)

Let me unpack the two axes people most often get wrong.

**Extrapolation is where the differences become dangerous, not just stylistic.** A GP with a stationary kernel does the *right* thing far from data: it reverts toward the mean function and its predictive interval *fans out*, loudly announcing "I don't know out here." A spline does the *wrong* thing quietly — past the boundary knots its basis functions go flat or linear, so it emits a confident-looking straight extrapolation that is a pure artifact of the basis. BART is worse still: outside the training range every tree just returns its nearest leaf's constant, so BART extrapolates as a **flat line with deceptively narrow bands**. If your application ever evaluates the model outside the training envelope — forecasting, dose extrapolation, covariate shift — this axis alone can decide the choice, and it argues for the GP.

**Smoothness is a *modeling assumption*, not a free lunch.** If you *know* the truth is smooth (physics, biology, a slowly-drifting trend), encoding that with a spline or GP is a feature — it denoises and regularizes for you. If the truth genuinely has jumps or thresholds (a regime change, a policy cutoff, a hard saturation), BART's bumpiness becomes an *advantage* and the GP/spline will wrongly smear the discontinuity. Match the model's smoothness to the world's smoothness.

> 🔧 **In practice:** These are not mutually exclusive. Combine them: a GP for the smooth temporal trend *plus* BART for the messy covariate effects; a spline for a known-smooth seasonal term inside a larger GLM. And when you genuinely can't decide, **fit two and compare with `az.compare` / LOO** (Chapter 11) — let predictive performance on held-out data break the tie, the way it should. (Reminder from §2.3: each `idata` must carry the pointwise log-likelihood first — sample with `idata_kwargs={"log_likelihood": True}` or call `pm.compute_log_likelihood(idata)` — or `az.compare` will refuse to run.) The decision table narrows the field; cross-validation settles the final.

> 📜 **Citation/Origin:** The deep equivalences here are real and worth knowing: a spline with a smoothing penalty *is* a GP with a particular kernel (the random-walk prior corresponds to a specific covariance), and a GP is the infinite-basis limit of a basis expansion. BART relates to ensembles/random forests with a Bayesian prior. BMCP Chapters 5 and 7 develop the spline and BART pictures; Rasmussen & Williams, *Gaussian Processes for Machine Learning*, Chapter 6, develops the spline↔GP equivalence. You are not learning three unrelated tricks — you are learning three coordinates on one map.

---

## ⚠️ Common errors & how to fix them

| Symptom | Cause | Fix |
|---|---|---|
| Spline fit is wiggly / overfits | Independent-Normal weight prior too loose, or too many knots | Tighten the weight `sigma`; better, switch to the `GaussianRandomWalk` smoothing prior on `w` so `tau` is *learned*; reduce knots |
| `ValueError`/shape mismatch building `mu = a + B @ w` | `dims`/`size` of `w` ≠ `B.shape[1]` (hardcoded the knot count) | Always set the weight dimension from `B.shape[1]`; the column count is `#interior_knots + degree + 1`, not `num_knots` |
| Spline matmul slow, or a copy-warning | `B` is row-major (C-order) from `patsy` | `B = np.asarray(B, order="F")` before passing to `pm.math.dot` |
| Basis looks wrong (humps piled at one end / flat-zero columns) | Passed full knot list incl. boundaries to `bs(knots=...)`, or knots outside data range | Pass **interior** knots only: `knots=knot_list[1:-1]`; place knots at `np.quantile(x, ...)` |
| Spline `tau` funnel → divergences | RW innovation scale near zero pinches geometry | Raise `target_accept` to 0.95–0.99; non-center the random walk — `z = pm.Normal('z',0,1,dims='splines'); w = pm.Deterministic('w', tau*pt.cumsum(z))` — as shown in §2.4 (same trick as Ch 10) |
| BART model errors / negative rates into Poisson | Forgot the link wrap — BART output is on the *link* scale | Pass `Y=np.log(Y)` and wrap `mu=pm.math.exp(mu)` for counts; `pm.math.invlogit(mu)` for Bernoulli |
| BART run buried in $\hat R$/ESS convergence warnings | Per-tree params aren't smooth NUTS targets; checks are noise on them | `pm.sample(..., compute_convergence_checks=False)`; judge by PPC and held-out accuracy, not tree-internal $\hat R$ |
| `pmb.plot_pdp` / `compute_variable_importance` raises on a keyword | `pymc-bart` helper API drifted between releases | `# verify in current pymc-bart docs`; check the live API reference for the current signature |
| BART PDP looks flat but the variable is important | PDP averages over interactions; the effect only appears jointly | Use a 2-D PDP, or `pmb.plot_ice` to see per-observation curves and heterogeneity |
| DP-mixture: many divergences, low ESS, weak $\hat R$ on `mu` | Inherent multimodality + label switching + concentration funnel | Expected for DP mixtures; raise `target_accept`, lengthen `tune`, standardize data; judge the **density**, not the labels |
| `cumprod`/`concatenate` import errors in stick-breaking | Old notebook uses `theano.tensor` / `aesara` | Use `import pytensor.tensor as pt`; `pt.extra_ops.cumprod`, `pt.concatenate` |
| DP mixture: last-index sticks carry real weight | Truncation `K` too small — you chopped off occupied clusters | Raise `K` and refit; verify tail weights ≈ 0 via `az.plot_forest(idata, var_names=["w"])` |
| Spline/BART extrapolates confidently and wrongly past the data | Neither extrapolates honestly; basis goes flat, trees return edge leaves | Don't extrapolate with these; use a GP (Ch 12) or structural time series (Ch 15) when you must predict outside the data envelope |

---

## 🧪 Exercises

1. **(Conceptual) "Nonparametric" ≠ "no parameters."** In two or three sentences, explain to a skeptical colleague why a 30-component truncated DP mixture (with well over a hundred parameters) is called *nonparametric*, while a degree-3 polynomial (4 parameters) is *parametric*. *Hint: the distinction is about whether model capacity is fixed in advance or grows with the data.*

2. **(Coding — knots & overfitting) ** Take the synthetic 1-D data from §1. Fit the **independent-Normal** spline at `num_knots ∈ {5, 15, 40}` and plot all three posterior-mean curves with their HDI bands. Then refit at `num_knots = 40` using the **`GaussianRandomWalk` smoothing prior** and overlay it. Describe what happens to wiggliness as knots increase under the independent prior, and how the smoothing prior rescues the 40-knot fit. *Hint: report the posterior of `tau` — what value does the data choose, and what does that imply about smoothness?*

3. **(Coding — BART link scale) ** Deliberately introduce the classic BART bug: fit the bikes model passing `Y` (not `np.log(Y)`) to `pmb.BART` and using `mu=mu` (no `exp`) in the NegBin. What error or pathology appears? Now fix it. In one paragraph, explain *why* BART must operate on the link scale and the likelihood must do the inverse-link wrap — connect this back to the GLM link-function logic of **Chapter 09**.

4. **(Coding — interpretation) ** On the fitted bikes BART model, produce (a) the variable-importance plot and (b) a 2-D partial-dependence plot of `hour × temperature`. Interpret: at what hours and temperatures is rental demand highest? Then produce `pmb.plot_ice` for `hour` and identify at least one way the per-observation curves disagree with the averaged PDP. *Hint: weekday vs weekend riders may have different hour profiles — does the ICE reveal two populations?*

5. **(Coding — DP mixture & truncation) ** Fit the truncated stick-breaking DP mixture to the Old Faithful `waiting`-time data (`sns.load_dataset("geyser")["waiting"]`, standardized). Run it at `K = 5`, `K = 15`, and `K = 30`. Use `az.plot_forest` on `w` to check, for each `K`, whether the tail sticks are empty. What is the smallest `K` that is clearly "enough"? Overlay the posterior-predictive density on a histogram of the data — how many modes does the DP mixture find, and does it match what you'd guess by eye?

6. **(Judgment — the decision table) ** For each scenario, name the tool (spline / GP / BART) you'd reach for *first* and justify it in one sentence using the §5 axes: (a) forecasting monthly atmospheric CO₂ five years into the future with calibrated uncertainty; (b) predicting customer churn from 40 mixed numeric/categorical features; (c) estimating a smooth dose-response curve from 60 observations of a single drug dose; (d) modeling an outcome with a known hard threshold at a policy cutoff. *Hint: scenario (a) hinges on the extrapolation axis; (d) hinges on the smoothness axis.*

---

## 📚 Resources & further reading

**Bayesian splines**
- **PyMC — Splines (how-to):** https://www.pymc.io/projects/examples/en/latest/howto/spline.html — the verified backbone: `patsy` `dmatrix(bs(...))`, `np.asarray(B, order="F")`, `pm.math.dot`. Copy its skeleton.
- **Bayesian Modeling and Computation in Python (Martin, Kumar, Lao) — Ch 5, Splines:** https://bayesiancomputationbook.com/markdown/chp_05.html — B-splines, the random-walk/P-spline smoothing prior, free and the PyMC/ArviZ companion to this course.
- **McElreath, *Statistical Rethinking* 2e — Ch 4 (cherry blossoms):** https://github.com/rmcelreath/rethinking — the gentlest introduction to splines as basis functions; PyMC port in `pymc-devs/pymc-resources`.
- **Eilers & Marx, *"Flexible smoothing with B-splines and penalties"* (Statistical Science 11(2):89–121, 1996):** https://projecteuclid.org/journals/statistical-science/volume-11/issue-2/Flexible-smoothing-with-B-splines-and-penalties/10.1214/ss/1038425655.full (DOI: 10.1214/ss/1038425655) — the original P-spline paper behind the difference-penalty smoothing prior.

**BART**
- **PyMC-BART — Introduction example:** https://www.pymc.io/projects/bart/en/latest/examples/bart_introduction.html — verified coal (Poisson) and bikes (NegBin) BART models, PDP and variable-importance calls.
- **PyMC-BART — API reference:** https://www.pymc.io/projects/bart/en/latest/api_reference.html — verify `pmb.BART(...)`, `plot_pdp`, `plot_ice`, `compute_variable_importance`, `plot_variable_importance` signatures (the moving target — check before relying on a keyword).
- **PyMC-BART — GitHub repo:** https://github.com/pymc-devs/pymc-bart — install, versions, `split_rules` details.
- **Quiroga, Garay, Alonso, Loyola & Martin, *"Bayesian additive regression trees for probabilistic programming"* (2022):** https://arxiv.org/abs/2206.03619 — the `pymc-bart` paper: BART-as-PPL-primitive, the PGBART sampler, the depth-penalizing prior.
- **Bayesian Modeling and Computation in Python — Ch 7, BART:** https://bayesiancomputationbook.com/markdown/chp_07.html — BART chapter by the `pymc-bart` authors; intuition + diagnostics.
- **Chipman, George & McCulloch, *"BART: Bayesian Additive Regression Trees"* (Annals of Applied Statistics 4(1):266–298, 2010):** https://projecteuclid.org/journals/annals-of-applied-statistics/volume-4/issue-1/BART-Bayesian-additive-regression-trees/10.1214/09-AOAS285.full (DOI: 10.1214/09-AOAS285) — the original method, including the depth-penalizing prior $\alpha(1+d)^{-\beta}$.

**Dirichlet process / infinite mixtures**
- **PyMC — Dirichlet process mixtures for density estimation:** https://www.pymc.io/projects/examples/en/latest/mixture_models/dp_mix.html — verified truncated stick-breaking with `pt.extra_ops.cumprod`, `alpha ~ Gamma`, truncation checks. Direct source for §4.
- **PyMC — Dependent density regression:** https://www.pymc.io/projects/examples/en/latest/mixture_models/dependent_density_regression.html — DP mixtures as flexible *regression*; a natural extension.
- **PyMC API — `pm.NormalMixture`:** https://www.pymc.io/projects/docs/en/latest/api/distributions/generated/pymc.NormalMixture.html — per-component `mu`/`sigma`/`tau` semantics used in the stick-breaking model.

**GP↔spline↔BART connections**
- Rasmussen & Williams, *Gaussian Processes for Machine Learning* (2006), Ch 6 — the spline↔GP equivalence (free PDF at gaussianprocess.org/gpml).
- Cross-reference **Chapter 12 (Gaussian Processes)** of this course for kernels, HSGP/sparse approximations, and honest extrapolation — the third corner of the §5 triangle.

---

## ➡️ What's next

You have now seen how to let inference choose a function's *shape* (splines, BART) and a mixture's *size* (the Dirichlet process). The common thread — a flexible thing unrolling in some dimension, tied together by a prior that controls how fast it's allowed to change — turns out to be the exact machinery you need for the dimension we have so far ignored: **time**. A Gaussian random walk on spline weights is a smoothing prior; a Gaussian random walk on *the state of a system over time* is a state-space model. **Chapter 15 — Time Series & State-Space Models** picks up the random walk you met in §2.4 and runs with it: autoregressive models, structural decompositions of trend and seasonality, stochastic volatility, change-points, and forecasting with the honest, widening uncertainty that — as §5 warned you — splines and BART could never give. Bring the random-walk intuition; you already have the hardest part.
