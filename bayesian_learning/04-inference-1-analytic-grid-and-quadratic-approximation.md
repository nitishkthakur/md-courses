# Chapter 04 — Inference I: Analytic, Grid & Quadratic (Laplace) Approximation

### *How to actually compute a posterior — by hand, by brute force, and by a clever Gaussian shortcut — and exactly when each one lies to you.*

> *A note from me to you.*
>
> *Up to now we've been doing the romantic part of Bayes: writing down generative
> stories, choosing priors, drawing the owl. But there is a quiet, unglamorous
> question lurking under every model we've built — **how do you turn the symbol
> $p(\theta \mid y)$ into actual numbers you can plot, summarize, and act on?**
> Bayes' theorem hands you the posterior up to a constant, and that constant — the
> evidence $p(y)$ — is an integral that, for almost any model with more than one or
> two parameters, you cannot do with pencil, paper, or even a very patient computer.*
>
> *This chapter is where we confront that wall for the first time. We'll start with
> the two cases where the wall isn't there at all (conjugate analytic posteriors,
> already in your pocket from Chapter 02), then climb it three ways: by **gridding**
> the parameter space and just computing the posterior at every point (beautiful,
> exact-ish, and doomed the moment you have more than ~3 parameters), and by the
> **quadratic / Laplace approximation** — the trick of pretending the posterior is a
> Gaussian centered at its peak. That last idea is McElreath's `quap`, it is the
> mathematical seed of variational inference, and it is the reason "just take the
> inverse Hessian" appears in a thousand papers. I want you to leave this chapter
> able to do all three by hand, to feel in your bones **why** the curse of
> dimensionality kills the grid, and — most importantly — to recognize the specific
> shapes of posterior where the Gaussian shortcut quietly gives you a confident,
> wrong answer. That recognition is what separates someone who runs `fit_laplace`
> and trusts the output from someone who knows when to throw it away and call NUTS.
> The painful mistake this chapter prevents is **trusting a tidy approximation on a
> posterior that was never going to be tidy.***

---

## What you'll be able to do after this chapter

- **Explain precisely why** the evidence integral $p(y)=\int p(y\mid\theta)p(\theta)\,d\theta$ is the computational bottleneck of Bayesian inference, and why conjugacy is the rare escape hatch.
- **Implement grid approximation** from scratch in NumPy for one- and two-parameter models, plot the posterior surface, and draw posterior samples by resampling the grid.
- **Demonstrate the curse of dimensionality** with real numbers — show how grid cost scales as $K^d$ and why that ends the grid party at 3–4 parameters.
- **Derive the Laplace / quadratic approximation** from a second-order Taylor expansion of the log-posterior, and state exactly why the covariance is the inverse negative Hessian at the mode.
- **Compute a Laplace approximation in PyMC v5** three ways: the one-call `pymc_extras.fit_laplace`, the explicit `find_MAP(compute_hessian=True)`, and a from-scratch manual Hessian via `model.compile_d2logp` — and know which symbols are version-fragile.
- **Diagnose when Laplace is excellent** (large $n$, unimodal, near-Gaussian — the Bernstein–von Mises regime) **and when it fails** (skew, boundaries, multimodality, hierarchical funnels, heavy tails), by comparing against an exact posterior and against MCMC.

---

## 1. Why we even need approximation: the integral in the denominator

Let me re-derive the thing that's about to ruin our day. Bayes' theorem for parameters $\theta$ given data $y$ is

$$
p(\theta \mid y) \;=\; \frac{p(y \mid \theta)\,p(\theta)}{p(y)}, \qquad
p(y) \;=\; \int p(y \mid \theta)\, p(\theta)\, d\theta .
$$

In ASCII, so it's burned in:

```
                 likelihood × prior
posterior  =  --------------------------
              ∫ likelihood × prior  dθ      ← "the evidence", a.k.a. the problem
```

The numerator is *easy*. You wrote the likelihood; you chose the prior; for any
specific $\theta$ you can evaluate $p(y\mid\theta)\,p(\theta)$ in a microsecond.
That product is the **unnormalized posterior**, and — this is worth tattooing
somewhere — **the unnormalized posterior already contains the entire shape of the
distribution.** Every relative height, every ridge and valley, every mode, is in
the numerator. The denominator $p(y)$ is just one number, a constant with respect
to $\theta$, whose only job is to make the whole thing integrate to 1.

So why is one constant such a big deal? Because to *get* that constant you must
integrate the numerator over **all** of parameter space, and you must do it well
enough that the normalized posterior is trustworthy. For a single parameter on a
bounded interval, that integral is a freshman calculus exercise — you can do it on
a grid in milliseconds. For two parameters it's a double integral, still fine. For
the hundreds-of-parameter hierarchical (radon) model you'll build in Chapter 10, it
is a several-hundred-dimensional integral over a space so vast that no grid, no
quadrature rule, and no amount of compute will ever touch more than an infinitesimal
speck of it.

> 💡 **Intuition:** The posterior is the numerator. Everything hard about Bayesian
> computation is the *normalizing constant* — and, more subtly, computing
> **expectations** under the posterior (means, intervals, predictive probabilities),
> which are *also* integrals over $\theta$. Inference is integration. Every method
> in the next three chapters — grid, Laplace, MCMC, VI — is a different strategy for
> dodging or approximating these integrals.

> 🧮 **The math: why dimension is the enemy.** Numerical quadrature on a regular
> grid needs the integrand evaluated at a lattice of points. If you place $K$ points
> along each of $d$ parameter axes, you have $K^d$ total evaluations. The *accuracy*
> of simple quadrature improves only polynomially in $K$, but the *cost* grows
> exponentially in $d$. With $K=100$ points per axis (modest!), one parameter is
> 100 evaluations; two is 10,000; five is ten billion; ten is $10^{20}$ — more grid
> points than there are grains of sand on Earth, to fit a model you'd consider
> *small*. This is the **curse of dimensionality**, and it is not a software
> limitation you can engineer around. It is geometry.

There are exactly two ways out, and the rest of this course is built on them:

1. **Be lucky in algebra** — pick a prior so compatible with the likelihood that the
   integral has a closed form. This is **conjugacy**, and it's wonderful when it
   applies and almost never applies to interesting models. (Chapter 02.)
2. **Approximate.** Either approximate the *integral* (grid / quadrature — this
   chapter, §3), approximate the *posterior with a tractable shape* (Laplace this
   chapter §4; variational inference, Chapter 06), or approximate *expectations by
   sampling* without ever computing $p(y)$ at all (MCMC, Chapter 05).

This chapter walks the first rungs of that ladder. Let me set the stage with the
lucky case before we start climbing.

---

## 2. The lucky case: analytic / conjugate posteriors (a recap with intent)

You met conjugacy in **Chapter 02 (Priors I)**, so I'll keep this brief but make the
*connection to computation* explicit, because that's the angle that matters here.

A prior is **conjugate** to a likelihood when the posterior lands in the same
distributional family as the prior. When that happens, the scary integral $p(y)$
solves itself — it's a known normalizing constant — and the posterior parameters
update by simple arithmetic. The canonical example, which will headline our grid
demo in a moment, is **Beta–Binomial**:

$$
\theta \sim \mathrm{Beta}(\alpha,\beta), \qquad
y \mid \theta \sim \mathrm{Binomial}(n,\theta)
\;\;\Longrightarrow\;\;
\theta \mid y \sim \mathrm{Beta}(\alpha + y,\; \beta + n - y).
$$

That's it. Six successes in nine globe tosses with a flat $\mathrm{Beta}(1,1)$ prior
gives a $\mathrm{Beta}(7,4)$ posterior — exactly, no integration, no sampling. We
know its mean ($\tfrac{7}{11}\approx 0.636$), its mode, its variance, its quantiles,
all in closed form. Other conjugate pairs you'll see: **Gamma–Poisson** (rate
parameters for counts), **Normal–Normal** (a Gaussian mean with known variance), and
**Normal-Inverse-Gamma** (a Gaussian mean *and* variance together).

> 📜 **Citation/Origin:** The full catalog of conjugate pairs and the precise update
> rules live in Chapter 02 and in BDA3 (Gelman et al., 2013) Chapters 2–3. We treat
> them there as *prior-choice* tools; here they matter as the **only** family of
> models where exact inference is free.

So why doesn't conjugacy save us? Three reasons, and they're worth stating plainly
because they motivate everything else:

- **Most useful models aren't conjugate.** A logistic regression, a Student-$t$
  robust regression, a hierarchical model with a half-normal hyperprior — none have a
  closed-form posterior. The moment you put a sensible prior on a sensible
  likelihood, conjugacy usually evaporates.
- **Conjugacy constrains your priors.** Sometimes the conjugate prior is a poor
  description of your actual beliefs, and you shouldn't contort your model just to
  win an algebra prize.
- **Even when it exists, you often want the *whole workflow*** — prior predictive
  checks, posterior predictive checks, model comparison — and those are far easier in
  a unified sampling framework than juggling special-case formulas.

> 🔧 **In practice:** Conjugacy still earns its keep in three places: (1) as
> exact ground truth to validate an approximation against — which is exactly how
> we'll use Beta–Binomial below; (2) inside Gibbs samplers, where conjugate
> *conditionals* give cheap exact updates (Chapter 05); and (3) as fast components of
> bigger models. Treat it as a precious special case, not a general method.

With the lucky case parked, let's build the first general-purpose tool: when you
can't integrate, **tabulate**.

---

## 3. Grid approximation: brute force, and why it's beautiful before it's doomed

The idea is so simple it feels like cheating. You can't do the integral
analytically? Fine — **chop parameter space into a fine grid, evaluate the
unnormalized posterior at every grid point, and normalize by summing.** No calculus.
Just arithmetic and a lot of function evaluations.

This is the very first inference algorithm McElreath teaches in *Statistical
Rethinking* (§2.4.3, the globe-tossing example), and it's the right place to start
because it makes the abstract concrete: you will literally *see* the posterior as a
table of numbers, and watch normalization turn likelihood×prior into a probability
distribution.

### 3.1 The recipe, in four steps

> 🧮 **The math.** For a one-parameter model with parameter $\theta$ on an interval:
>
> 1. **Define the grid.** Choose $K$ points $\theta_1,\dots,\theta_K$ spanning the
>    plausible range (for a probability, that's $[0,1]$).
> 2. **Evaluate prior × likelihood** at each grid point:
>    $u_k = p(y\mid\theta_k)\,p(\theta_k)$. This is the *unnormalized posterior*.
> 3. **Normalize.** Approximate the evidence by the grid sum (or trapezoidal rule):
>    $Z \approx \sum_k u_k\,\Delta\theta$ (with $\Delta\theta$ the spacing), then set
>    $p(\theta_k\mid y) \approx u_k / Z$.
> 4. **Use it.** You now have a discrete approximation to the posterior. Summaries
>    (mean, quantiles, intervals) are weighted sums; to get *samples* you resample
>    the grid points with replacement, weighted by their posterior probability.

ASCII version of the whole thing:

```
grid      = θ_1, θ_2, ..., θ_K         (K candidate parameter values)
unnorm[k] = likelihood(y | θ_k) * prior(θ_k)
posterior = unnorm / sum(unnorm)       (now sums to 1 — a proper pmf over the grid)
samples   = choice(grid, size=N, p=posterior)   (turn it into draws)
```

### 3.2 Worked example: the globe toss, end to end

The setup is McElreath's: you toss a globe, catch it, and record whether your right
index finger lands on **W**ater or **L**and. You want the posterior for $\theta$, the
true proportion of the globe covered by water. You observe **6 W in 9 tosses**. The
likelihood is $\mathrm{Binomial}(n=9,\theta)$ evaluated at $y=6$. We'll use a flat
$\mathrm{Beta}(1,1)$ prior so we can check the grid against the *exact* $\mathrm{Beta}(7,4)$
conjugate posterior.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)

# --- data ---
successes, n = 6, 9          # 6 W in 9 globe tosses

# --- step 1: the grid ---
K = 1000
grid = np.linspace(0, 1, K)               # candidate values of theta in [0, 1]

# --- step 2: prior x likelihood (unnormalized posterior) ---
prior = np.ones_like(grid)                # flat Beta(1,1): density = 1 everywhere
likelihood = stats.binom.pmf(successes, n=n, p=grid)   # P(y=6 | n=9, theta) at each grid point
unnorm_post = likelihood * prior

# --- step 3: normalize ---
# np.trapz integrates with the trapezoidal rule; newer NumPy renames it np.trapezoid.
posterior = unnorm_post / np.trapz(unnorm_post, grid)   # a proper density over the grid

# --- step 4: draw posterior samples by resampling the grid ---
# `posterior` is a DENSITY (trapz-normalized so the area = 1). For np.random.choice we
# instead need a sum-to-1 PMF *over the grid points*, so we renormalize by a plain sum.
# Two different normalizations on purpose: density for plotting, pmf for resampling.
probs = posterior / posterior.sum()       # density -> sum-to-1 weights over the K grid points
samples = rng.choice(grid, size=10_000, p=probs)
```

Two lines deserve a pause. First, `stats.binom.pmf(successes, n=n, p=grid)` is the
*whole likelihood function*, evaluated at a thousand candidate $\theta$ values at
once — vectorization is doing the heavy lifting. Second, the resampling step is the
bridge from "a curve" to "draws you can treat like MCMC output": once you have
`samples`, every downstream summary (means, 89% intervals, the probability that
$\theta > 0.5$) is just NumPy on an array, *exactly* as it will be when those draws
come from NUTS.

> 🩺 **Diagnostic — read the plot.** Plot `posterior` against `grid` and overlay the
> exact $\mathrm{Beta}(7,4)$ density. You should see two curves that lie essentially
> on top of each other: a smooth hump peaking near $\theta \approx 0.67$ (the MAP, the
> mode of the posterior), with most mass between roughly 0.4 and 0.85. The grid curve
> is the brute-force answer; the Beta curve is the exact one. Their agreement is the
> point — **grid approximation is "correct" up to grid resolution.** Now overlay a
> histogram of `samples` and confirm it traces the same hump; that's your sanity check
> that resampling preserved the distribution.

```python
# verify the grid against the exact conjugate posterior, Beta(1+6, 1+3) = Beta(7, 4)
exact = stats.beta(1 + successes, 1 + (n - successes))

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(grid, posterior, lw=3, label="grid posterior (K=1000)")
ax.plot(grid, exact.pdf(grid), "k--", lw=2, label="exact Beta(7,4)")
ax.hist(samples, bins=50, density=True, alpha=0.25, label="10k resampled draws")
ax.axvline(grid[np.argmax(posterior)], color="C3", ls=":",
           label=f"MAP ≈ {grid[np.argmax(posterior)]:.3f}")
ax.set_xlabel(r"$\theta$ (proportion water)"); ax.set_ylabel("density")
ax.legend(); ax.set_title("Globe toss: grid vs exact posterior")
```

Compute a couple of numbers so the agreement isn't just visual:

```python
post_mean = np.sum(grid * probs)                 # grid posterior mean
print("grid mean   :", round(post_mean, 4))      # ~0.6364
print("exact mean  :", round(exact.mean(), 4))   # 7/11 = 0.6364
print("P(theta>0.5):", round((samples > 0.5).mean(), 4))   # ~0.83  (exact: 1 - Beta(7,4).cdf(0.5) = 0.828)
# 89% interval (McElreath's habit: avoid the false comfort of "95")
print("89% interval:", np.quantile(samples, [0.055, 0.945]).round(3))
```

The grid mean matches the exact $7/11$ to four decimals. That's grid approximation
working perfectly — because in one dimension, with a thousand points, it *is*
essentially exact.

> ⚠️ **Pitfall:** Two grid-specific traps. (1) **Truncating the grid too tightly.**
> If your grid doesn't span the region where the posterior has mass, you silently
> renormalize a *truncated* distribution and get confidently wrong intervals. For
> bounded params like a probability this is safe; for an unbounded mean it is not —
> always sanity-check that the posterior has decayed to ~0 at both grid edges. (2)
> **Too few points** under-resolves sharp posteriors; the curve gets blocky and
> quantiles jitter. For 1-D, $K=1000$ is luxurious; the problem is never resolution
> in 1-D, it's *dimension*, which we hit next.

### 3.3 Two parameters: a posterior *surface*

Let's go to 2-D, both because real models have more than one parameter and because
it's the last dimension where you can still *see* the whole posterior. We'll fit a
Gaussian with **unknown mean $\mu$ and unknown standard deviation $\sigma$** to a
small synthetic sample — a model with no one-line conjugate posterior (the joint
posterior of $\mu$ and $\sigma$ together is not a standard named distribution), so
the grid is genuinely earning its keep.

```python
# --- synthetic data from a KNOWN generative process (so we can check recovery) ---
true_mu, true_sigma = 5.0, 2.0
y = rng.normal(true_mu, true_sigma, size=20)     # n = 20 observations

# --- a 2-D grid over (mu, sigma) ---
mu_grid    = np.linspace(2.0, 8.0, 200)
sigma_grid = np.linspace(0.5, 5.0, 200)
MU, SIGMA  = np.meshgrid(mu_grid, sigma_grid)    # 200 x 200 = 40,000 grid cells

# --- log unnormalized posterior (work in LOG space — products of many small numbers underflow) ---
# log likelihood: sum over data of Normal(y_i | mu, sigma) log-density, vectorized over the grid
def log_likelihood(mu, sigma):
    # broadcast: (grid..., 1) vs data (n,) -> sum over the data axis
    ll = stats.norm.logpdf(y[:, None, None], loc=mu[None, :, :], scale=sigma[None, :, :])
    return ll.sum(axis=0)

log_prior = (stats.norm.logpdf(MU, loc=0, scale=10)          # weakly-informative prior on mu
             + stats.expon.logpdf(SIGMA, scale=5))            # Exponential(1/5) prior on sigma > 0
log_unnorm = log_likelihood(MU, SIGMA) + log_prior

# --- normalize in a numerically safe way: subtract the max before exponentiating ---
log_unnorm -= log_unnorm.max()                   # now the peak is exp(0) = 1, no overflow
post2d = np.exp(log_unnorm)
post2d /= post2d.sum()                            # discrete normalization over the 40k cells
```

> 🔧 **In practice — always grid in log space.** A likelihood is a *product* of
> $n$ densities, each often $\ll 1$. Multiply twenty of them and you flirt with
> floating-point underflow; multiply two hundred and you hit zero exactly. So we sum
> **log**-densities, and only exponentiate at the very end — and even then we subtract
> the maximum first (`log_unnorm -= log_unnorm.max()`) so the largest value is
> $\exp(0)=1$. This "log-sum-exp" hygiene is not optional; it is the difference
> between a clean surface and an array of `NaN`s. You'll see the same trick inside
> every sampler and inside Stan's `target +=`.

Now visualize the surface and pull out marginals:

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# (a) the joint posterior as a filled contour / heatmap
cs = axes[0].contourf(MU, SIGMA, post2d, levels=25, cmap="viridis")
axes[0].plot(true_mu, true_sigma, "r*", ms=15, label="true (μ, σ)")
axes[0].set_xlabel(r"$\mu$"); axes[0].set_ylabel(r"$\sigma$")
axes[0].set_title("Joint posterior surface  p(μ, σ | y)"); axes[0].legend()

# (b) marginal posteriors by SUMMING OUT the other axis (= numerical marginalization)
post_mu    = post2d.sum(axis=0)     # integrate over sigma -> marginal of mu
post_sigma = post2d.sum(axis=1)     # integrate over mu    -> marginal of sigma
axes[1].plot(mu_grid, post_mu / post_mu.max(), label=r"marginal $p(\mu|y)$")
axes[1].plot(sigma_grid, post_sigma / post_sigma.max(), label=r"marginal $p(\sigma|y)$")
axes[1].axvline(true_mu, color="C0", ls=":"); axes[1].axvline(true_sigma, color="C1", ls=":")
axes[1].set_title("Marginal posteriors (summed over the other axis)"); axes[1].legend()
```

> 🩺 **Diagnostic — read the surface.** Panel (a) is a single blob of probability
> mass, roughly elliptical, centered near $(\mu,\sigma)\approx(5,2)$ — and the red
> star (the *true* values that generated the data) sits inside the bright core. That
> recovery is the reassurance that the machinery works. Notice the blob is slightly
> tilted and *asymmetric in $\sigma$*: it stretches farther toward large $\sigma$
> than small. That asymmetry is your first glimpse of why a Gaussian approximation
> (next section) will struggle with scale parameters. Panel (b) shows the
> **marginals**, obtained by literally summing the surface along one axis — and
> summing a discrete grid *is* numerical integration. The marginal of $\mu$ is
> near-symmetric; the marginal of $\sigma$ has a visible right skew (a longer tail
> toward larger values). Hold onto that skew.

> 💡 **Intuition:** Marginalizing — "integrating out" a nuisance parameter to see one
> parameter on its own — is just **summing the joint surface along that parameter's
> axis.** When people say "the posterior marginal of $\mu$," in grid-world that's one
> line of NumPy. In MCMC-world (Chapter 05) it's even easier: you just *ignore* the
> other columns of your draws. Same concept, two implementations.

### 3.4 The curse of dimensionality, with numbers that hurt

We've now done 1-D ($K=1000$ points, instant) and 2-D ($200\times200 = 40{,}000$
cells, still instant). Let me make the wall visceral. Suppose you insist on a
modest $K=100$ points per axis. Here's the grid cost as you add parameters:

| Parameters $d$ | Grid cells $K^d$ (with $K=100$) | Roughly… |
|---:|---:|---|
| 1 | $10^2$ | instant |
| 2 | $10^4$ | instant |
| 3 | $10^6$ | a blink (≈8 MB; float64 = 8 bytes/cell) |
| 4 | $10^8$ | seconds, ~0.8 GB if stored |
| 5 | $10^{10}$ | minutes, ~80 GB — won't fit in RAM |
| 6 | $10^{12}$ | ~8 terabytes of grid |
| 10 | $10^{20}$ | more cells than sand grains on Earth |
| ~few hundred (Ch. 10's radon model) | $\gg 10^{200}$ | more cells than atoms in the universe ($\approx 10^{80}$) — by hundreds of orders of magnitude |

```python
for d in [1, 2, 3, 4, 5, 6, 10]:
    print(f"d={d:>2}:  {100**d:.0e} grid cells")
# d= 1: 1e+02   d= 2: 1e+04   d= 3: 1e+06   d= 4: 1e+08
# d= 5: 1e+10   d= 6: 1e+12   d=10: 1e+20
```

> 🧮 **The math — and the deeper geometry.** The headline cost is $K^d$, exponential
> in dimension. But there's a *second*, subtler killer that grid sampling can't beat
> even if you had infinite memory: in high dimensions, **the posterior's mass is not
> near its mode.** It concentrates in a thin shell called the **typical set**, whose
> location and shape you don't know in advance, so a uniform grid wastes essentially
> all of its points in regions of negligible probability. (We'll make this precise in
> Chapter 05 — it's the single most important geometric fact behind why HMC exists.)
> So the grid doesn't just get *expensive* with dimension; it gets *misallocated*.
> Adaptive and sparse grids buy you a few dimensions; nothing buys you a few hundred.

This table is the entire pedagogical reason the rest of the course exists. Grid
approximation is the honest, exact-ish baseline you reach for in 1–2 dimensions to
*understand* a posterior or to *check* another method. Past 3–4 parameters it is
dead, and we need a method whose cost grows gracefully with dimension. The first such
method — cheap, deterministic, and the conceptual ancestor of variational inference —
is the **quadratic approximation**.

---

## 4. Quadratic / Laplace approximation: pretend it's a Gaussian

Here is the idea in one sentence: **near its peak, almost any smooth, unimodal
posterior looks like a Gaussian — so just find the peak and measure how sharply the
posterior curves away from it, and you've got a Gaussian that approximates the whole
thing.** That's the entire method. It's called the **Laplace approximation** (after
Pierre-Simon Laplace, who used it in the 1770s–1810s), and in McElreath's *Statistical
Rethinking* it goes by the friendly name **`quap` — quadratic approximation**. The
two words are the two ingredients: *quadratic*, because we approximate the log-posterior
by a quadratic (a parabola in 1-D, a paraboloid in $d$-D); *approximation*, because
that quadratic is exact only at the peak and degrades as you move away.

> 💡 **Intuition:** Why a Gaussian, of all shapes? Because the **logarithm of a
> Gaussian is exactly a parabola** — $\log \mathcal{N}(\theta\mid m, s^2)$ is
> $-\tfrac{1}{2s^2}(\theta - m)^2 + \text{const}$, a downward parabola peaked at $m$.
> So "approximate the posterior by a Gaussian" is *the same statement* as "approximate
> the **log**-posterior by a parabola." And approximating any smooth function near its
> peak by a parabola is just the second-order Taylor expansion — the most natural
> approximation in all of calculus. The Gaussian isn't an arbitrary choice; it's what
> you get automatically when you Taylor-expand a log-density to second order.

### 4.1 The derivation (do not skip this — it's three lines and it explains everything)

Write the log of the unnormalized posterior as
$\ell(\theta) = \log\big[p(y\mid\theta)\,p(\theta)\big]$. We don't need the
normalizing constant: it only shifts $\ell$ up or down by a constant, which won't
affect the *shape* we're about to extract. Let $\hat\theta$ be the **mode** of the
posterior — the value that maximizes $\ell$. Because it's the posterior mode, it is
also the **MAP** (maximum a posteriori) estimate. Now Taylor-expand $\ell$ around
$\hat\theta$ to second order:

$$
\ell(\theta) \;\approx\; \ell(\hat\theta)
\;+\; \underbrace{\nabla\ell(\hat\theta)^\top (\theta - \hat\theta)}_{=\,0 \text{ at the mode}}
\;-\; \tfrac{1}{2}\,(\theta - \hat\theta)^\top H\, (\theta - \hat\theta),
\qquad
H \;=\; -\,\nabla^2 \ell(\theta)\big|_{\hat\theta}.
$$

Look at the three terms. The **constant** $\ell(\hat\theta)$ is just the peak height —
it folds into normalization and we drop it. The **linear** term vanishes *exactly*,
because at a maximum the gradient is zero ($\nabla\ell(\hat\theta) = 0$) — that's the
definition of $\hat\theta$. So the *first surviving* term is the **quadratic** one,
governed by $H$, the **negative Hessian** of the log-posterior at the mode. ($H$ is the
matrix of second derivatives, negated so it comes out positive-definite at a maximum —
the posterior curves *downward* away from a peak, so the raw Hessian is negative.)

Now exponentiate to get back to the posterior:

$$
p(\theta\mid y) \;\propto\; e^{\ell(\theta)} \;\approx\;
e^{\ell(\hat\theta)}\,\exp\!\Big(-\tfrac{1}{2}(\theta - \hat\theta)^\top H\,(\theta - \hat\theta)\Big).
$$

That right-hand side is — up to the constant — *exactly* the kernel of a multivariate
Gaussian with mean $\hat\theta$ and covariance $H^{-1}$. So:

$$
\boxed{\;p(\theta\mid y) \;\approx\; \mathcal{N}\!\big(\hat\theta,\; H^{-1}\big),
\qquad H = -\nabla^2 \log p(\theta\mid y)\big|_{\hat\theta}.\;}
$$

In ASCII, the sentence to remember:

```
posterior ≈ Normal( mean = MAP,  covariance = inverse of the negative Hessian
                                              of the log-posterior at the MAP )
```

> 🧮 **The math — what the Hessian *means*.** The Hessian is curvature. A
> log-posterior that is *sharply* peaked (large second derivative, large $H$) curves
> down fast — the data pinned the parameter down — and its inverse $H^{-1}$ is small,
> i.e. **small posterior variance, high certainty**. A log-posterior that is *flat*
> near the top (small $H$) gives a large $H^{-1}$, i.e. **wide posterior, low
> certainty**. So $H^{-1}$ as the covariance isn't a formula to memorize — it's the
> precise statement that *curvature is information* and *width is uncertainty*. In 1-D
> it collapses to the familiar $\sigma^2 \approx -1/\ell''(\hat\theta)$: posterior
> standard deviation is one over the square root of the curvature. (Frequentists will
> recognize $H$ as the observed Fisher information — same matrix, and that's not a
> coincidence; it's the bridge in the Bernstein–von Mises theorem below.)

So the Laplace recipe is just two computations:

1. **Find the mode** $\hat\theta$ — an *optimization* problem (maximize $\ell$), which
   scales to high dimensions beautifully (gradient ascent / L-BFGS), unlike the grid.
2. **Compute the curvature** $H$ at the mode and invert it to get the covariance.

That's it. No integral. The cost is one optimization plus one Hessian — cheap,
deterministic, and it returns a full joint Gaussian posterior approximation, complete
with correlations between parameters. This is why Laplace is the natural next rung
above the grid.

### 4.2 A footgun before we write any code

> ⚠️ **Pitfall — the `pm.Laplace` name clash.** In core PyMC, `pm.Laplace` is a
> **probability distribution** — the double-exponential (a.k.a. the Laplace
> distribution), used as a likelihood or a sparsity-inducing prior. It has **nothing
> to do** with the Laplace *approximation* we just derived. And there is **no
> `pm.quap`** — `quap` is a function in McElreath's *R* package `rethinking`, not in
> PyMC. So if you go looking for `pm.quap` or expect `pm.Laplace` to approximate a
> posterior, you'll be confused. The PyMC-native tools for the *approximation* live in
> the **`pymc_extras`** package (formerly `pymc-experimental`), which we use next.

### 4.3 Laplace in PyMC v5 — the recommended one-call API

The cleanest way to get a Laplace approximation in modern PyMC is
`pymc_extras.fit_laplace`. It finds the MAP, computes the inverse Hessian at the
mode, builds the Gaussian, draws samples from it, and hands you back an
ArviZ-compatible object so all your usual `az.*` tools just work. Let's apply it to
the **Howell !Kung height** dataset — McElreath's canonical `quap` example — fitting
a simple linear regression of height on weight. This is *exactly* the regime where
Laplace shines (lots of data, a well-identified near-Gaussian posterior), so we expect
it to nearly nail the answer.

```python
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import pymc_extras as pmx          # formerly pymc-experimental; `import pymc_extras as pmx` is current

RANDOM_SEED = 8927

# --- load the Howell !Kung data and keep adults (height is ~linear in weight for adults) ---
url = "https://raw.githubusercontent.com/rmcelreath/rethinking/master/data/Howell1.csv"
d = pd.read_csv(url, sep=";")
adults = d[d.age >= 18].copy()

# --- standardize the predictor (course default; sets up sane priors — see Ch 3) ---
w = adults.weight.values
w_std = (w - w.mean()) / w.std()           # mean 0, sd 1
height = adults.height.values

coords = {"obs": np.arange(len(height))}
with pm.Model(coords=coords) as howell_model:
    weight = pm.Data("weight", w_std, dims="obs")     # mutable container for later prediction
    a = pm.Normal("a", mu=178, sigma=20)              # intercept: mean adult height (cm)
    b = pm.Normal("b", mu=0, sigma=10)                # slope: cm of height per 1 SD of weight
    sigma = pm.HalfNormal("sigma", 25)                # residual SD (PyMC uses SD, not variance/precision!)
    mu = a + b * weight
    pm.Normal("height", mu=mu, sigma=sigma, observed=height, dims="obs")

# --- the Laplace approximation, in ONE call ---
with howell_model:
    idata_laplace = pmx.fit_laplace(
        optimize_method="BFGS",        # any scipy.optimize method; BFGS is the default
        draws=4000,                    # samples drawn from the fitted Gaussian
        random_seed=RANDOM_SEED,
        # include_transformed=True,    # also report params on the unconstrained scale
    )   # -> returns an arviz DataTree / InferenceData-like object; az.* works on it
```

> 🔧 **In practice — versions move; pin your spelling.** `pymc_extras` is
> fast-moving. The signature above matches the 0.9.x/0.10.x docs (the `fit_laplace`
> reference page is in §Resources), but **verify the argument names against the
> version you have installed** — `# verify in current pymc_extras docs`. The **direct
> function is the safe call**: `from pymc_extras.inference import fit_laplace`. (The
> high-level `pmx.fit(method="pathfinder", ...)` wrapper is documented for *Pathfinder*
> in Chapter 06; don't assume a `method="laplace"` dispatch exists — call `fit_laplace`
> directly.) The dossier's documented default is `draws=500`; we bump it to `draws=4000`
> here purely to get a smooth Gaussian for the plots — it's an explicit choice, not a
> requirement.

Now treat the result like any posterior:

```python
print(az.summary(idata_laplace, var_names=["a", "b", "sigma"]))
# approximate — your values will land within rounding (packages aren't pinned; the
# optimizer + sampled draws vary slightly run to run):
#         mean     sd   hdi_3%  hdi_97%   ...
# a     154.6   0.27   154.1    155.1     (≈ mean adult height in cm)
# b       5.8   0.28     5.3      6.3     (≈ +5.8 cm per 1 SD of weight)
# sigma   5.1   0.19     4.7      5.4     (residual scatter, cm)
az.plot_posterior(idata_laplace, var_names=["a", "b", "sigma"])
```

> 🩺 **Diagnostic — read the output.** `az.summary` gives you the Gaussian's mean and
> SD for each parameter plus a 94% HDI (highest-density interval). The slope `b` is
> comfortably positive and tight — heavier people are taller, ~5.8 cm per standard
> deviation of weight, with little uncertainty because we have hundreds of adults.
> `az.plot_posterior` will show three clean bell curves; that's *guaranteed* here,
> because Laplace **literally returns a Gaussian** — the marginals *cannot* be
> anything but bell-shaped. (Read that twice: the smoothness of these plots is a
> property of the method, **not evidence that the approximation is good.** We'll
> verify goodness by comparing to MCMC in §5.)

### 4.4 Laplace the explicit way — `find_MAP(compute_hessian=True)`

When you want the MAP point *and* the covariance matrix in your hands — say to
inspect parameter correlations, or to seed a sampler — use the two-step form. The key
flag is `compute_hessian=True`; without it you get only the point.

```python
from pymc_extras.inference import find_MAP

with howell_model:
    map_res = find_MAP(
        method="L-BFGS-B",
        compute_hessian=True,          # <-- this is what makes it Laplace, not just MAP
        gradient_backend="pytensor",
        random_seed=RANDOM_SEED,
    )   # returns a DataTree with the MAP and the inverse-Hessian covariance

# A MvNormal(mean = MAP, cov = inverse_Hessian) IS the Laplace approximation.
# (Exact attribute names for the covariance vary by version — # verify in current docs.)
```

> ⚠️ **Pitfall — stock `pm.find_MAP` is NOT Laplace.** Core PyMC's `pm.find_MAP()`
> (no `pymc_extras`) returns **only a point-estimate dictionary** — the MAP, full
> stop. No Hessian, no covariance, no uncertainty. On its own it gives you a *point*,
> not a *distribution*. To turn it into a Laplace approximation you must add the
> curvature yourself, which is exactly what `compute_hessian=True` or the manual route
> below does. Reporting `pm.find_MAP` as "the posterior" is a classic beginner error:
> it throws away all the uncertainty, which is the entire point of being Bayesian.

### 4.5 Laplace by hand — demystifying the Hessian with `compile_d2logp`

To make sure the machinery isn't magic, let's build the Laplace covariance ourselves
from PyMC's compiled second-derivative function. PyMC can hand you a callable for the
Hessian of the model's log-probability via `model.compile_d2logp()`. The recipe:
optimize to the MAP, evaluate the Hessian of the **negative** log-posterior there,
and invert it.

```python
import numpy as np

with howell_model:
    # 1) find the mode (MAP). pm.find_MAP returns a dict keyed by the model's FREE RVs
    #    on their UNCONSTRAINED scale — here that's "a", "b", and "sigma_log__"
    #    (sigma is positive, so PyMC optimizes log(sigma), not sigma).
    map_point = pm.find_MAP(method="L-BFGS-B")            # dict of MAP values

    # 2) compile the Hessian (matrix of 2nd derivatives) of the model logp.
    hess_fn = howell_model.compile_d2logp()               # callable: point -> Hessian of logp
    # (API/return shape can vary by version — # verify in current PyMC docs)

# 3) Evaluate the Hessian at the MAP. compile_d2logp wants a full point dict over ALL
#    free RVs on the unconstrained scale. Start from the model's template point and
#    overwrite each entry with the MAP value, so keys/shapes line up exactly.
point = howell_model.initial_point()                      # template: correct keys & shapes
point = {k: np.asarray(map_point[k]) for k in point}      # plug in MAP values for the SAME keys

# The Hessian of logp is NEGATIVE-definite at a max; the negative-Hessian H is
# positive-definite, and its INVERSE is the Laplace covariance (full joint, with correlations).
H = -np.asarray(hess_fn(point))             # negative Hessian of logp -> positive-definite
cov_laplace = np.linalg.inv(H)              # <-- THE Laplace covariance (3x3 here)

# Order of rows/cols matches the order of the free RVs in `point`:
labels = list(point.keys())                 # e.g. ["a", "b", "sigma_log__"]
map_vector = np.array([np.asarray(point[k]).ravel()[0] for k in labels])
print("free RVs (unconstrained):", labels)
print("MAP vector              :", map_vector.round(3))   # note: sigma is log(sigma) here
print("Laplace covariance:\n", cov_laplace.round(4))
print("Laplace SDs             :", np.sqrt(np.diag(cov_laplace)).round(3))
# The SD on the "sigma_log__" row is the SD of log(sigma); exponentiate draws to get sigma.
# Exact key spelling / ordering / which params are transformed are model- and version-specific.
# verify in current PyMC docs
```

> 🧮 **The math you just executed.** Those few lines *are* the boxed equation from
> §4.1. `find_MAP` gave you $\hat\theta$; `compile_d2logp` gave you $\nabla^2\ell$;
> negating gives $H$; inverting gives $H^{-1}$. The result is the **full $3\times3$**
> covariance over the model's free RVs (`a`, `b`, and `sigma_log__` — `sigma` on the
> log scale). The diagonal of $H^{-1}$ holds the per-parameter posterior variances
> (take the square root for SDs); the off-diagonals are the posterior *covariances*
> between parameters — which is how Laplace captures correlations the mean-field VI of
> Chapter 06 will throw away. One honest detail to internalize: the third row/column is
> $\log\sigma$, not $\sigma$ — its SD is the spread of $\log\sigma$, and you recover
> `sigma` by exponentiating draws, exactly as `fit_laplace` does for you under the hood.
> Everything `fit_laplace` did automatically, you just did by hand.

> ⚠️ **Pitfall — the scale is unconstrained, and that's a feature.** Positive
> parameters like `sigma` cannot be Gaussian on their natural scale — a Gaussian puts
> mass below zero, which is illegal for a standard deviation. PyMC sidesteps this by
> optimizing and computing the Hessian on the **unconstrained (log) scale**: `sigma`
> is handled as $\log\sigma \in \mathbb{R}$, where a Gaussian is perfectly legal, and
> the draws are back-transformed (exponentiated) afterward. This is *why* QUAP can
> look decent for a scale parameter even though the natural-scale posterior is skewed —
> the Gaussian lives in log-space, and McElreath's `quap` does the identical trick.
> The flip side: the reported MAP for `sigma` is the mode *on the log scale,
> back-transformed*, which is **not** the natural-scale posterior mean. Don't compare
> them naively. (This subtlety is exactly why we cross-check against MCMC next.)

---

## 5. When Laplace is great, and when it lies — with proof

A method that always returns a smooth Gaussian is dangerous precisely because the
output *looks* trustworthy whether or not it is. So we need a clear mental model of
**when the Gaussian-at-the-mode is a good approximation** and when it isn't — and then
we'll *prove* both cases by overlaying Laplace against the gold-standard MCMC posterior
(Chapter 05's NUTS, which we'll lean on as ground truth throughout the rest of the
course).

### 5.1 When Laplace works: the Bernstein–von Mises regime

> 📜 **Citation/Origin — Bernstein–von Mises.** There's a deep theorem behind why
> Laplace works so often in practice. Loosely: **as the amount of data grows, the
> posterior of a well-behaved, identified model converges to a Gaussian** centered at
> the true parameter, with covariance shrinking like $1/n$ (the inverse Fisher
> information). This is the **Bernstein–von Mises theorem**, the Bayesian cousin of the
> central limit theorem. Its practical reading: *with enough data and a regular model,
> the posterior eventually becomes Gaussian — which is exactly the shape Laplace
> assumes.* So Laplace isn't a crude hack; in the large-$n$ limit it becomes
> **asymptotically exact.** The art is knowing how much data is "enough" and whether
> your model is "regular."

Laplace is excellent when **all** of these roughly hold:

- **Lots of data relative to parameters** (the Bernstein–von Mises regime). The Howell
  regression above — hundreds of adults, three parameters — is a poster child.
- **Unimodal posterior.** One peak, no competing modes.
- **Roughly Gaussian on the (transformed) scale** — symmetric, light tails, no hard
  boundary cutting into the mass.
- **Well-identified parameters** — the data actually constrain every parameter (the
  Hessian is well-conditioned, not near-singular).

In that regime, Laplace gives you a near-exact posterior in a tiny fraction of the
cost of MCMC, and it's a *fantastic* tool: for fast model iteration, for initializing
a sampler, for back-of-envelope uncertainty.

### 5.2 Proof it works: Laplace vs MCMC on Howell

Let's earn our trust. Refit the Howell model with NUTS and lay the two posteriors on
top of each other.

```python
with howell_model:
    idata_nuts = pm.sample(
        1000, tune=1000, chains=4, target_accept=0.9,
        random_seed=RANDOM_SEED,    # returns InferenceData
    )

# overlay Laplace (the Gaussian approx) against NUTS (ground truth) for each parameter
az.plot_forest(
    [idata_nuts, idata_laplace],
    model_names=["NUTS (truth)", "Laplace"],
    var_names=["a", "b", "sigma"],
    combined=True,
)
# numeric side-by-side:
print(az.summary(idata_nuts,    var_names=["a", "b", "sigma"])[["mean", "sd"]])
print(az.summary(idata_laplace, var_names=["a", "b", "sigma"])[["mean", "sd"]])
```

> 🩺 **Diagnostic — read the forest plot.** `az.plot_forest` draws each parameter's
> point estimate and interval as a horizontal whisker, one row per method. For Howell
> you'll see the NUTS and Laplace whiskers **essentially coincide** — same centers,
> same widths — for `a`, `b`, and even `sigma`. That overlap is the visual proof that
> in the large-$n$, unimodal, well-identified regime, **Laplace ≈ the exact
> posterior**. The only place you might catch daylight is the *upper* edge of
> `sigma`'s interval, where the true posterior has a hair more right-skew than a
> Gaussian allows — a tiny preview of the failure mode we're about to provoke
> deliberately.

### 5.3 When Laplace lies: the catalog of failure shapes

Now the important half. Laplace fails — sometimes catastrophically, always silently —
whenever the posterior is *not* well-described by a single Gaussian. Memorize these
shapes; each one breaks the approximation in a characteristic way:

| Failure shape | What goes wrong | Why |
|---|---|---|
| **Skewed / asymmetric** posterior | Gaussian is symmetric → wrong intervals; tail width mis-estimated | The quadratic can't bend; it symmetrizes a lopsided peak |
| **Bounded parameter near a boundary** (e.g. a probability with little data, $\theta$ near 0 or 1) | Gaussian leaks mass past the boundary; mode pinned at the edge has ill-defined curvature | The posterior is truncated; a parabola isn't |
| **Multimodal** posterior | Laplace grabs **one** mode and ignores all the others | Taylor expansion is local — it never sees the other peaks |
| **Heavy tails** (Student-$t$-like) | Gaussian tails decay too fast → underestimates extreme-value probability | Curvature at the mode says nothing about a fat tail far away |
| **Hierarchical funnel** (variance params, the Eight-Schools neck) | Laplace badly misses the narrow neck where $\tau \to 0$; geometry is non-Gaussian *everywhere* | The posterior curls into a funnel no single Gaussian can fit |
| **Strong nonlinear correlations** | A single ellipse can't trace a banana-shaped ridge | The Gaussian's contours are ellipses; the truth is curved |
| **Small $n$ / strong prior** | Posterior simply hasn't reached the Bernstein–von Mises regime yet | Asymptotics haven't kicked in |

> ⚠️ **Pitfall — the silent failure.** Every one of these still produces a clean,
> confident-looking Gaussian summary. `fit_laplace` will not warn you that it grabbed
> the wrong mode or symmetrized a skewed tail; `az.plot_posterior` will draw a serene
> bell curve regardless. **The only reliable check is to compare against MCMC** (or, in
> 1–2 dims, against a grid). This is why the rest of this course treats NUTS as ground
> truth and treats *any* fast approximation as provisional until validated.

### 5.4 Proof it lies: a bounded probability with tiny data

Let's force a clean, visible failure with the simplest possible model — back to the
globe toss, but with **almost no data and an extreme outcome** so the posterior piles
up against the boundary at $\theta = 1$. Observe **3 W in 3 tosses**: every toss was
water. The true posterior (with a flat prior) is $\mathrm{Beta}(4,1)$, which is *heavily*
right-skewed, monotonically increasing toward $\theta = 1$, with its mode pinned at the
boundary. A symmetric Gaussian has no chance of capturing this.

```python
import pymc as pm, arviz as az, numpy as np
from scipy import stats
import pymc_extras as pmx

RANDOM_SEED = 8927
successes, n = 3, 3          # 3 water in 3 tosses — extreme, tiny data

with pm.Model() as boundary_model:
    theta = pm.Beta("theta", 1, 1)                     # flat prior on [0, 1]
    pm.Binomial("y", n=n, p=theta, observed=successes)

# Laplace approximation (the Gaussian-at-the-mode):
with boundary_model:
    idata_lap = pmx.fit_laplace(draws=4000, random_seed=RANDOM_SEED)

# Ground truth #1: MCMC.  Ground truth #2: the exact Beta(4,1) conjugate posterior.
with boundary_model:
    idata_mcmc = pm.sample(2000, tune=1000, chains=4, target_accept=0.9,
                           random_seed=RANDOM_SEED)
exact = stats.beta(1 + successes, 1 + (n - successes))   # Beta(4, 1)

# Compare all three on the natural [0,1] scale:
grid = np.linspace(0, 1, 500)
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.plot(grid, exact.pdf(grid), "k-", lw=3, label="exact Beta(4,1)")
# NOTE: az.plot_kde has no top-level `label`; the legend text goes INSIDE plot_kwargs.
az.plot_kde(idata_mcmc.posterior["theta"].values.flatten(), ax=ax,
            plot_kwargs={"color": "C0", "lw": 2, "label": "NUTS"})
az.plot_kde(idata_lap.posterior["theta"].values.flatten(), ax=ax,
            plot_kwargs={"color": "C3", "lw": 2, "ls": "--", "label": "Laplace (Gaussian)"})
ax.set_xlim(0, 1.2); ax.set_xlabel(r"$\theta$"); ax.legend()
ax.set_title("3 W in 3 tosses: Laplace cannot follow a boundary-pinned, skewed posterior")
```

> 🩺 **Diagnostic — read the failure.** Three curves, and now they *disagree loudly*.
> The exact $\mathrm{Beta}(4,1)$ (black) and the NUTS posterior (blue) agree: both rise
> monotonically toward $\theta = 1$, all the mass crammed against the boundary, no
> symmetric peak at all. The **Laplace** curve (red dashed) does something nonsensical:
> it places a *symmetric bell* whose center sits below 1 and whose **right half spills
> past $\theta = 1$ into impossible territory** ($\theta > 1$ is not a probability).
> It badly underestimates the mass near 1 and invents mass where none can exist. If you
> reported the Laplace 94% interval here, it would include values above 1. This is the
> boundary failure in full color — and it's *invisible* unless you draw this comparison.

> 💡 **Intuition — why this specific break.** Two things conspire. First, the posterior
> is **monotone**, so its "mode" is at the boundary $\theta = 1$, where the curvature
> the Hessian needs is ill-defined (the peak isn't an interior peak). Second, even after
> the logit transform PyMC uses to handle the $[0,1]$ constraint, with only three data
> points the transformed posterior is far from Gaussian — we are nowhere near the
> Bernstein–von Mises regime. Add data (say 30 W in 30 tosses) and even this would
> Gaussianize on the logit scale and Laplace would recover. *Small data near a boundary*
> is the lethal combination.

> 🔧 **In practice — your standing Laplace checklist.** Before you trust a Laplace /
> QUAP posterior, ask: (1) Do I have **lots of data** relative to parameters? (2) Is the
> posterior plausibly **unimodal**? (3) Are any parameters **bounded and near a
> boundary**, **heavy-tailed**, or part of a **hierarchical variance** structure? (4)
> Can I afford a **cheap MCMC or grid cross-check**? If you can't answer (1)–(2) "yes"
> and (3) "no," reach for NUTS. Laplace is a scalpel for a specific surgery, not a
> general-purpose tool — and the rest of the course is about the general-purpose tools.

---

## 6. Where this leaves us: the bridge to MCMC and VI

Step back and look at the ladder we've climbed. Each rung trades exactness for
scalability in a different way:

| Method | Cost in $d$ dims | Exact? | Captures | Dies when |
|---|---|---|---|---|
| **Analytic / conjugate** | free | exact | everything | the model isn't conjugate (almost always) |
| **Grid** (§3) | $K^d$ — *exponential* | up to resolution | the full shape, any number of modes | $d \gtrsim 3$–$4$ (curse of dimensionality) |
| **Laplace / QUAP** (§4–5) | one optimization + one Hessian | only near-Gaussian unimodal posteriors | a single Gaussian blob *with* correlations | skew, boundaries, multimodality, funnels |
| **MCMC / NUTS** (Ch 05) | scales to thousands of dims | asymptotically exact (no shape assumption) | *anything* — multimodal, skewed, funnels | slow / expensive per effective sample |
| **Variational inference** (Ch 06) | fast optimization | approximate (mode-seeking, too narrow) | a chosen tractable family | reverse-KL underestimates variance |

> 💡 **Intuition — the two escape routes from the integral, and where each goes
> next.** Grid and Laplace are both *deterministic* attacks on the posterior: grid
> samples parameter space on a lattice and dies of dimension; Laplace replaces the
> posterior with a Gaussian it can write down in closed form and dies of non-Gaussian
> shape. The next two chapters are the two ways to escape *both* limitations.
>
> **Chapter 05 (MCMC, from Metropolis to NUTS)** keeps Laplace's "explore parameter
> space" idea but makes it *stochastic* and *shape-agnostic*: instead of one
> optimization to a single mode, you run a Markov chain that wanders the posterior in
> proportion to its density, so it can trace skew, multiple modes, and funnels — no
> Gaussian assumption at all. Crucially it never computes $p(y)$; it needs only the
> *unnormalized* posterior (the easy numerator from §1), which is exactly why it works
> where the grid couldn't. That's the gold standard, and it's why we leaned on NUTS as
> ground truth throughout this chapter.
>
> **Chapter 06 (Variational inference)** keeps Laplace's "replace the posterior with a
> tractable distribution" idea but makes it *smarter*: instead of fixing the Gaussian at
> the mode via the Hessian, VI *optimizes* the parameters of an approximating family
> $q(\theta)$ to be as close as possible — in KL divergence — to the true posterior. The
> Laplace approximation is, in fact, a crude special case of this idea. You'll see that
> VI inherits Laplace's congenital weakness — it tends to return **the right shape of a
> too-narrow distribution** — and you'll learn to *detect* that with the very move we
> used here: compare to NUTS. You'll also meet **Pathfinder**, a modern method that runs
> an optimizer like Laplace but harvests a *sequence* of Gaussians along the optimization
> path and importance-resamples them — the "best of both worlds" successor to QUAP.

So Laplace isn't a dead end; it's the conceptual seed of half of what's coming. The
"find the mode, measure the curvature" reflex you built here is *literally* how modern
samplers initialize and how Pathfinder operates. Master it as a tool *and* as an idea.

---

## ⚠️ Common errors & how to fix them

| Symptom | Cause | Fix |
|---|---|---|
| `AttributeError: module 'pymc' has no attribute 'quap'` | `quap` is an *R* `rethinking` function; there is no `pm.quap` | Use `pymc_extras.fit_laplace` or `pymc_extras.find_MAP(compute_hessian=True)` |
| You used `pm.Laplace` and got a *distribution*, not an approximation | `pm.Laplace` is the **double-exponential distribution**, unrelated to the Laplace *approximation* | Import the approximation from `pymc_extras`; never confuse the two |
| `pm.find_MAP()` gives a point but no uncertainty | Stock core `pm.find_MAP` returns **only** the MAP dict — no Hessian/covariance | Use `pymc_extras.find_MAP(compute_hessian=True)` or `fit_laplace`; or build the Hessian via `model.compile_d2logp()` and invert |
| Laplace covariance has negative variances / Hessian singular or non-positive-definite | Mode at a boundary, a flat ridge, or a non-identified / nearly-collinear parameter | Standardize predictors, add weakly-informative priors, reparameterize (non-centered), or abandon Laplace for NUTS |
| Laplace 94% interval for a probability includes values `> 1` (or `< 0`) | Symmetric Gaussian leaks past a hard boundary — the §5.4 failure | Don't use Laplace near a boundary with little data; use the exact conjugate posterior, a grid, or NUTS |
| Laplace posterior far too narrow / misses a second peak | Quadratic Taylor expansion is **local** — it captures one mode only | Multimodal posterior ⇒ use NUTS; Laplace is the wrong tool |
| Grid posterior looks blocky / quantiles jitter | Too few grid points under-resolve a sharp posterior | Increase $K$ (cheap in 1-D); but if $d \geq 3$, switch methods entirely |
| Grid posterior confidently wrong / intervals truncated | Grid range doesn't span where the posterior has mass | Widen the grid until density decays to ≈0 at both edges; check the tails |
| `RuntimeWarning: overflow` / `NaN`s building the grid surface | Multiplying many small likelihood values under/overflows | Work in **log space**, subtract the max before `exp` (log-sum-exp), as in §3.3 |
| Reported `sigma` MAP ≠ the natural-scale posterior mean | MAP is computed on the **unconstrained (log) scale** and back-transformed; mode ≠ mean under a transform | Expected — compare *distributions*, not the bare MAP, to the natural-scale posterior |
| `ImportError: cannot import name 'fit_laplace'` | Package renamed (`pymc-experimental` → `pymc_extras`) or moved submodule | `import pymc_extras as pmx`; try `from pymc_extras.inference import fit_laplace`; `# verify in current docs` |

---

## 🧪 Exercises

**1. (Conceptual — the integral.)** In one or two sentences each: (a) Why is the
*numerator* of Bayes' theorem easy but the *denominator* $p(y)$ hard? (b) Conjugacy
makes $p(y)$ trivial — give one concrete reason you still wouldn't rely on conjugacy for
general modeling. (c) The curse of dimensionality is $K^d$; explain in your own words why
*adaptive* grids buy only a couple of extra dimensions rather than solving the problem.
*(Hint: the typical set, §3.4.)*

**2. (Grid — coding.)** Redo the globe-toss grid approximation of §3.2, but replace the
flat prior with an informative $\mathrm{Beta}(2,2)$ prior (a gentle preference for
$\theta$ near 0.5). (a) Plot the new grid posterior against the exact conjugate
posterior — what distribution is it now? (b) Recompute the posterior mean and 89%
interval. (c) Vary $K \in \{10, 50, 1000\}$ and overlay the curves: at what resolution
does the grid become visually indistinguishable from exact? *(Hint: the conjugate update
is $\mathrm{Beta}(\alpha + y,\, \beta + n - y)$.)*

**3. (Grid — the curse, by hand.)** Without writing code, compute the number of grid
cells for a model with $d = 7$ parameters at $K = 100$ points per axis, and at $K = 50$.
Then estimate the memory to store the posterior surface in float64 (8 bytes/cell). State
the largest $d$ you could grid on a 32 GB machine at $K = 50$. *(Hint: $\log_{10}$
arithmetic; bytes $= 50^d \times 8$.)*

**4. (Laplace — recover the truth.)** Generate synthetic data from a logistic regression
with a *known* coefficient vector (say $\beta = [-0.5, 1.2]$, $n = 500$). Fit it in PyMC,
then approximate the posterior three ways: `pymc_extras.fit_laplace`, the manual-Hessian
route (§4.5), and `pm.sample` (NUTS). (a) Confirm all three recover $\beta$. (b) Overlay
them with `az.plot_forest`. (c) This is a large-$n$, well-identified GLM — explain, citing
Bernstein–von Mises, *why* Laplace should and does succeed here. *(Hint: standardize the
predictors first; use `pm.Bernoulli` with a logit link.)*

**5. (Laplace — provoke a failure.)** Take your model from Exercise 4 and shrink the data
to $n = 8$ with strong class imbalance (e.g. 7 zeros, 1 one). (a) Fit Laplace and NUTS.
(b) Plot the marginal of one coefficient with `az.plot_kde` for both. (c) Describe the
disagreement: which failure shape from the §5.3 table are you seeing? *(Hint: small $n$ +
near-separable data pushes a coefficient toward $\pm\infty$ — a heavy/skewed tail.)*

**6. (Synthesis — choose the tool.)** For each scenario, state which method from this
chapter (analytic / grid / Laplace) you'd reach for *first*, and one sentence why:
(a) a single Beta–Binomial proportion; (b) estimating two parameters $(\mu,\sigma)$ of a
Normal from 30 points for a teaching plot; (c) a 5-parameter regression you want a *fast*
uncertainty estimate on before committing to a full MCMC run; (d) a hierarchical model
with a variance hyperparameter you suspect produces a funnel. *(Hint: (d) is a trap — name
the right escape hatch even though it isn't in this chapter.)*

---

## 📚 Resources & further reading

**Books & course material**
- McElreath, *Statistical Rethinking* (2nd ed., 2020), **§2.4.3** (grid approximation,
  the globe toss) and **§4.3–4.4** (the canonical `quap` height regression on the Howell
  data). The conceptual source for this entire chapter; the PyMC port lives in
  [`pymc-devs/pymc-resources`](https://github.com/pymc-devs/pymc-resources).
- Martin, Kumar & Lao, *Bayesian Modeling and Computation in Python* —
  [bayesiancomputationbook.com](https://bayesiancomputationbook.com), **Ch 11**
  ("what's under the hood"): Laplace, grid, and VI with idiomatic PyMC/ArviZ.
- Gelman, Carlin, Stern, Dunson, Vehtari & Rubin, *Bayesian Data Analysis* (BDA3, 2013):
  **Ch 4** on the normal/large-sample (Laplace) approximation and modal approximations;
  **Ch 2–3** for the conjugate updates referenced in §2. Free PDF online.

**The `quap` / Laplace lineage**
- McElreath's `quap` reference — [rdrr.io/github/rmcelreath/rethinking/man/quap.html](https://rdrr.io/github/rmcelreath/rethinking/man/quap.html)
  and source [github.com/rmcelreath/rethinking/blob/master/R/map-quap.r](https://github.com/rmcelreath/rethinking/blob/master/R/map-quap.r)
  — confirms `quap` = MAP (mode) + quadratic curvature (Hessian); renamed from `map` in
  the 2nd edition.

**PyMC APIs (verify against your installed version — `pymc_extras` moves fast)**
- `pymc_extras.fit_laplace` — [pymc.io/projects/extras/.../fit_laplace.html](https://www.pymc.io/projects/extras/en/stable/generated/pymc_extras.inference.fit_laplace.html)
  — the recommended one-call Laplace; exact signature and defaults. The docs explicitly
  warn it is unsuitable for skewed/multimodal posteriors.
- `pymc_extras.find_MAP` — [pymc.io/projects/extras/.../find_MAP.html](https://www.pymc.io/projects/extras/en/stable/generated/pymc_extras.inference.find_MAP.html)
  — `compute_hessian=True` is the explicit Laplace ingredient.
- ArviZ `summary` & diagnostics — [python.arviz.org/.../arviz.summary.html](https://python.arviz.org/en/stable/api/generated/arviz.summary.html)
  — works directly on Laplace/MCMC InferenceData for the side-by-side comparisons.
- PyMC Discourse, **"Laplace approximation vs VI"** — [discourse.pymc.io/t/laplace-approximation-vs-vi/17722](https://discourse.pymc.io/t/laplace-approximation-vs-vi/17722)
  — practitioner guidance on when to reach for each.

**Theory: why Laplace works (and its successors)**
- The **Bernstein–von Mises theorem** — covered in BDA3 §4 and van der Vaart,
  *Asymptotic Statistics* (1998), Ch 10 — the asymptotic-Gaussianity result behind §5.1.
- Zhang, Carpenter, Gelman & Vehtari (2022), **"Pathfinder: Parallel quasi-Newton
  variational inference,"** JMLR 23(306) — [jmlr.org/papers/v23/21-0889.html](https://jmlr.org/papers/v23/21-0889.html)
  (arXiv:2108.03782). The modern descendant of QUAP; previews Chapter 06.
- Blei, Kucukelbir & McAuliffe (2017), **"Variational Inference: A Review for
  Statisticians"** — [arxiv.org/abs/1601.00670](https://arxiv.org/abs/1601.00670) — for
  the deeper "approximate the posterior with a tractable family" view that Laplace seeds.

**Dataset**
- Howell !Kung census — `pd.read_csv(".../rethinking/master/data/Howell1.csv", sep=";")`
  from the [`rmcelreath/rethinking`](https://github.com/rmcelreath/rethinking) repo.

---

## ➡️ What's next

We've squeezed every drop out of the *deterministic* approaches: exact algebra when we're
lucky, brute-force grids until dimension murders them, and the Gaussian-at-the-mode
shortcut until the posterior refuses to be Gaussian. The honest conclusion is that for the
models you actually care about — non-conjugate, high-dimensional, skewed, multimodal,
hierarchical — **none of these will do.** We need a method that needs only the easy
*unnormalized* posterior, makes **no assumption about the posterior's shape**, and scales
to thousands of parameters. That method is **Markov chain Monte Carlo**.

In **Chapter 05 — Inference II: MCMC from Metropolis to NUTS**, we build it from the
ground up: Markov chains and why a chain can "wander in proportion to the posterior,"
Metropolis–Hastings and Gibbs, the geometry of the typical set (the deeper reason the grid
failed in §3.4), and then the modern engine — Hamiltonian Monte Carlo and the No-U-Turn
Sampler that powers `pm.sample`. You already met NUTS in this chapter as our "ground
truth"; next we open the box and see exactly how it earns that title.
