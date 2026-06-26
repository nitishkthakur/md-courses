# Chapter 07 — MCMC Diagnostics & Debugging

### How to know when your sampler is lying to you — and how to fix it

> *A note from me to you.*
>
> *Here is the uncomfortable truth that separates people who use Bayesian methods from people who can be **trusted** with them: `pm.sample()` almost always returns something. It returns numbers, it returns a tidy table, it returns smooth-looking posteriors. Most of the time those numbers are fine. But some of the time — and exactly the times that matter most, the hard hierarchical models, the funnels, the heavy tails — those numbers are quietly, confidently **wrong**. The sampler explored the easy part of the posterior, never reached the hard part, and reported back as if everything were settled.*
>
> *This chapter is the immune system of the whole course. Everything before it taught you to **build** models and to understand **how** HMC and NUTS work. This chapter teaches you to **interrogate** what came back — to look a fitted model in the eye and decide whether to believe it. We will go through every diagnostic ArviZ gives you, one at a time, and I will show you not just how to compute it but how to **read** it: what a healthy one looks like, what a sick one looks like, and what the sickness means geometrically.*
>
> *And then we will do the single most important hands-on skill in applied Bayesian computation: take a model that is broken — the famous Eight Schools, fit in its naive "centered" form, throwing divergences and bad $\hat R$ — and **fix it** by reparameterizing, watching every diagnostic go from red to green. If you internalize one thing from this entire course, let it be the centered-to-non-centered move. It is the move you will reach for, again and again, for the rest of your Bayesian life. Let's earn it.*

---

## What you'll be able to do after this chapter

- **Run the full ArviZ diagnostic loop** on any fitted model and decide, with defensible thresholds, whether to trust it.
- **Read $\hat R$ correctly** — understand that modern $\hat R$ is the rank-normalized split-$\hat R$ of Vehtari et al. (2021), why the old $1.1$ threshold is too loose, and what a value above $1.01$ is actually telling you.
- **Distinguish bulk-ESS from tail-ESS**, know why you need both, and use MCSE to decide how many digits of a posterior estimate are real.
- **Read trace, rank, and forest plots** — recognize the "fuzzy caterpillar," know why rank plots are the more honest modern replacement for trace plots, and spot a stuck chain.
- **Understand divergences geometrically** — why a fixed step size fails where curvature is high, how to **localize** divergences with `az.plot_pair`, and why the funnel is the archetypal cause.
- **Perform the centered → non-centered reparameterization** on a hierarchical model from memory, and prove to yourself that divergences vanish and ESS jumps.
- **Read the energy plot and BFMI**, recognize a heavy-tail energy problem, and know the tuning recipe (raise `target_accept`, lengthen `tune`, adapt a full mass matrix, then raise `max_treedepth`).
- **Navigate the `InferenceData` object** — know what lives in `posterior`, `sample_stats`, `posterior_predictive`, `prior`, and `observed_data`.
- **Apply a master symptom → cause → fix table** to debug any sampling problem you hit in the wild.

Throughout, I assume you have met HMC and NUTS in **Chapter 05 (MCMC: From Metropolis to NUTS)** — leapfrog integration, momentum, the typical set, `target_accept`, the mass matrix. Chapter 05 introduced divergences and energy as *concepts*; here we go deep on **reading and acting on** them. We will use the canonical **Eight Schools** dataset (see the dataset catalog) as the worked example, exactly as the field has for forty years.

```python
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)
```

---

## 1. Why diagnostics exist: the sampler can converge to the wrong thing

Let me set the philosophical stage, because the diagnostics only make sense if you understand what they are protecting you from.

When we run MCMC we are making a bet. The bet is that a finite Markov chain — a few thousand correlated draws — is a good enough stand-in for the true posterior $p(\theta \mid y)$ that we can compute expectations from it:

$$
\mathbb{E}_{p}[f(\theta)] \;\approx\; \frac{1}{N}\sum_{n=1}^{N} f(\theta_n).
$$

The **ergodic theorem** promises this average converges to the truth — *in the limit*, *if* the chain is irreducible and aperiodic and has the posterior as its stationary distribution. But "in the limit" is doing enormous work. In practice we have a finite chain, and two things can go wrong, both invisible if you only look at the posterior summary:

1. **The chain hasn't reached stationarity** (it's still in transient warmup-like behavior, or wandering between modes). Then your draws are not from $p(\theta\mid y)$ at all — they're from wherever the chain happens to be.
2. **The chain has reached stationarity but is exploring it pathologically slowly** — or worse, there is a region of the posterior it **systematically cannot enter** because the geometry defeats the integrator. Then your draws are biased: you're averaging over the part of the posterior the sampler could reach, not the whole thing.

> 💡 **Intuition:** A convergence diagnostic is not a proof of correctness. It is a **smoke detector**. A clean diagnostic does not guarantee your fit is right — there are pathologies no diagnostic catches. But a *dirty* diagnostic almost always means something is genuinely wrong. So the rule is asymmetric and humbling: **dirty diagnostics are a hard stop; clean diagnostics are a green light to proceed, not a certificate.** We never "prove convergence." We **fail to find evidence of non-convergence**, which is the best a finite computation can do.

> 📜 **Citation/Origin:** This honest framing is the heart of Betancourt's *"Towards a Principled Bayesian Workflow"* and the diagnostic philosophy baked into Stan and ArviZ. The diagnostics in this chapter are not optional polish; in the Bayesian workflow (Chapter 08) they are a mandatory gate every model must pass before you interpret a single number.

Here's the part that makes Bayesian computation so much more trustworthy than it has any right to be: **HMC and NUTS give you diagnostics that random-walk samplers cannot.** A Metropolis sampler can be hopelessly biased and look perfectly healthy — it will happily report tight, smooth posteriors while completely missing a mode. HMC, because it integrates Hamiltonian dynamics, *complains* when the geometry defeats it. **Divergences** are HMC raising its hand to say "I could not integrate the dynamics here; I'm probably missing something." Betancourt calls them *truth-tellers*. The whole reason we prefer NUTS is not just speed — it's that NUTS tells you when it failed.

So our job in this chapter is to become fluent in that vocabulary of complaints.

---

## 2. The `InferenceData` object: where the evidence lives

Before we read diagnostics, we need to know where they live. When you call `pm.sample(...)` in PyMC v5, it returns an `az.InferenceData` object. This is **not** a NumPy array of draws — it's a structured, self-describing container built on `xarray`, organized into named **groups**.

> ⚠️ **Pitfall:** Old PyMC v3 returned an object called a `trace` (a `MultiTrace`). You will still see "trace" used loosely in blog posts and even in function names like `az.plot_trace`. In this course we say **InferenceData**, never "trace," except to note the old name once — which I just did. The object is much richer than a bag of samples, and treating it as one throws away most of the diagnostics.

The groups you will actually use:

| Group | What's in it | Dims |
|---|---|---|
| `posterior` | The posterior draws of every free + deterministic parameter | `(chain, draw, *param_dims)` |
| `sample_stats` | Per-draw NUTS diagnostics: `diverging`, `energy`, `lp`, `tree_depth`, `acceptance_rate`, `step_size`, `n_steps`, … | `(chain, draw)` |
| `posterior_predictive` | Replicated data $\tilde y$ simulated from the posterior (after `sample_posterior_predictive`) | `(chain, draw, *obs_dims)` |
| `prior` / `prior_predictive` | Draws from the prior and prior-predictive (after `sample_prior_predictive`) | `(chain, draw, …)` |
| `observed_data` | The actual $y$ you conditioned on | `(*obs_dims)` |
| `constant_data` | Fixed inputs (predictors, indices) | `(…)` |
| `log_likelihood` | Pointwise log-likelihood $\log p(y_i\mid\theta)$ — needed for LOO (Chapter 11) | `(chain, draw, *obs_dims)` |

The object is `xarray`-backed, which means you slice it by **name**, not by integer position — this is why we always pass `coords=` to `pm.Model`. A few moves you'll use constantly:

```python
# The boolean divergence flag for every draw of every chain:
idata.sample_stats["diverging"]                 # DataArray (chain, draw), dtype bool

# Count divergences across all chains:
n_div = int(idata.sample_stats["diverging"].sum())

# Grab one parameter's draws from one chain:
idata.posterior["mu"].sel(chain=0)

# The list of groups present:
idata.groups()                                   # -> ['posterior', 'sample_stats', ...]
```

> 🩺 **Diagnostic:** The very first line of every diagnostic loop is `int(idata.sample_stats["diverging"].sum())`. If that number is not zero, you have a geometry problem and you stop and fix it before reading anything else. We'll see why this single integer is so load-bearing in §7.

> 🔧 **In practice:** `sample_stats` only exists for gradient-based samplers (NUTS/HMC). If you sampled a discrete parameter with a Metropolis-family stepper, the energy/divergence machinery doesn't apply to it — those diagnostics are about the Hamiltonian integrator. The convergence diagnostics ($\hat R$, ESS) still apply to *every* parameter regardless of sampler.

---

## 3. $\hat R$ — the convergence statistic you must read first

### The idea: chains that agree have converged

You ran **multiple chains** (we always run `chains=4`) starting from **different, jittered initial points** (that's what `init="jitter+adapt_diag"`, the default, does — see Chapter 05). Here is the logic: if every chain has forgotten where it started and is now wandering around the same stationary distribution, then the chains should be **statistically indistinguishable**. The spread of values *within* a single chain should match the spread of values *across* chains. If, on the other hand, the chains are still in different places — stuck near different modes, or still drifting — then the between-chain spread will be **larger** than the within-chain spread, and that mismatch is detectable.

That is the entire idea behind the Gelman–Rubin statistic $\hat R$ (R-hat): it is, in essence, the ratio of total variance (within + between) to within-chain variance.

> 🧮 **The math:** With $M$ chains each of length $N$, let $W$ be the average **within-chain** variance and $B$ the **between-chain** variance. The classic statistic is
> $$
> \hat R \;=\; \sqrt{\frac{\frac{N-1}{N}\,W \;+\; \frac{1}{N}\,B}{W}}.
> $$
> If the chains agree, $B \approx W$, the numerator collapses to $W$, and $\hat R \to 1$. If between-chain variance dominates, $\hat R > 1$. ASCII form:
> ```
>            ((N-1)/N) * W  +  (1/N) * B
> R_hat = sqrt( --------------------------- )
>                          W
> ```

### Why the *modern* $\hat R$ is different (and why you must use it)

The classic statistic above has two failure modes that the modern version fixes. **You should know this, because it's why the threshold moved from $1.1$ to $1.01$.**

1. **It's blind to non-stationarity *within* a chain.** A chain that drifts steadily upward has inflated within-chain variance $W$, which can mask the problem and pull $\hat R$ back toward 1 — exactly when you should be alarmed. **Fix: split each chain in half** and treat the halves as separate chains. Now a within-chain trend shows up as a between-half disagreement. This is **split-$\hat R$**.
2. **It assumes finite variance.** If a marginal is heavy-tailed (think a scale parameter with a Cauchy-ish posterior), the variances $W$ and $B$ are dominated by extreme draws and the statistic becomes unreliable. **Fix: rank-normalize** — replace each draw by its rank in the pooled sample, then map ranks through the inverse normal CDF. This is monotone (it preserves convergence information) but gives every marginal nice light tails, so the variance-based statistic behaves. This is **rank-normalized $\hat R$**.

And one more refinement: rank-normalized split-$\hat R$ as described senses differences in **location** between chains. To also catch differences in **scale/tails**, you compute a second $\hat R$ on the **folded** draws (absolute deviations from the median) and report the **maximum** of the two. ArviZ does all of this for you.

> 📜 **Citation/Origin:** This is Vehtari, Gelman, Simpson, Carpenter, and Bürkner (2021), *"Rank-normalization, folding, and localization: An improved $\hat R$ for assessing convergence of MCMC"*, *Bayesian Analysis* 16(2):667–718. It is exactly what `az.rhat(idata, method="rank")` computes and what the `r_hat` column of `az.summary` reports. **The old folklore threshold of $\hat R < 1.1$ is far too loose** — the paper shows you can have serious bias at $1.1$. The course threshold, following the paper, is:

> 🩺 **Diagnostic — the $\hat R$ rule:** **$\hat R \le 1.01$ is good.** Anything above $1.01$ means the chains disagree more than chance allows — they have **not** converged to a common distribution. Possible causes, roughly in order of how often you'll see them: (a) **too little tuning/sampling** (the cheap fix — sample longer); (b) a **pathological geometry** the sampler can't traverse (the funnel — reparameterize); (c) genuine **multimodality** where chains are stuck near different modes (the model is poorly identified — rethink it). A value like $1.3$ is a screaming alarm; a value like $1.02$ is "sample a bit more and look at the rank plot." Never interpret a posterior with $\hat R > 1.01$.

> ⚠️ **Pitfall:** **$\hat R$ requires at least two chains.** With `chains=1` the `r_hat` column comes back `NaN` — there's nothing to compare. This is one excellent reason we always sample `chains=4`. You will also see `NaN` for constants and for `Deterministic` nodes that are exactly fixed; that's harmless.

We'll see real $\hat R$ values — good and bad — in the worked example in §9. For now, internalize: **scan the `r_hat` column first; every value must be $\le 1.01$.**

---

## 4. Effective sample size: bulk-ESS, tail-ESS, and MCSE

$\hat R$ tells you *whether* the chains agree. **ESS tells you how much independent information you actually have.** These are different questions, and a model can pass one and fail the other.

### Autocorrelation eats your sample size

MCMC draws are **correlated** — each draw is a small step from the last. If draw $n$ is nearly identical to draw $n-1$, then 1000 draws don't carry 1000 draws' worth of information; they carry fewer. The **effective sample size** quantifies how many *independent* draws your correlated chain is worth:

$$
\mathrm{ESS} \;=\; \frac{N}{1 + 2\sum_{t=1}^{\infty}\rho_t},
$$

where $\rho_t$ is the autocorrelation at lag $t$. If the draws were independent, all $\rho_t = 0$ and $\mathrm{ESS} = N$. The more autocorrelated the chain, the larger the sum, the smaller the ESS.

> 💡 **Intuition:** ESS is the honest currency. You might have drawn 4000 samples, but if they're sticky you might have, effectively, 200 independent ones. Every posterior estimate's precision is governed by ESS, not by the raw draw count. This is why a sampler that mixes well (low autocorrelation) is so much more valuable than one that just runs more iterations — NUTS gets you high ESS *per draw* because it makes long, decorrelated jumps along the typical set.

### Bulk-ESS vs tail-ESS — why ArviZ reports both

Here is a subtlety the classic single ESS misses, and it matters enormously for interval estimates.

- **Bulk-ESS** is computed on rank-normalized draws and measures how well the chain explores the **center** of the distribution — the reliability of the **mean and median**.
- **Tail-ESS** is the minimum of the ESS computed at the **5% and 95% quantiles** — the reliability of the **tails**, i.e. of your credible-interval endpoints.

A chain can mix beautifully in the bulk and terribly in the tails. Think of a heavy-tailed posterior: the sampler spends most of its time in the dense center (great bulk-ESS) and only occasionally ventures into the tails, where it then gets stuck (poor tail-ESS). If you report a 94% HDI from such a fit, the **endpoints** are unreliable even though the **mean** is fine. You need both numbers.

> 🩺 **Diagnostic — the ESS rule:** Aim for **ESS_bulk and ESS_tail $\gtrsim 400$** for stable estimates. The rule of thumb behind this is **roughly $\ge 100$ effective draws per chain** (Vehtari et al. give a stricter floor of $\ge 50$/chain; with 4 chains, $\ge 400$ total is the comfortable headline number this course uses). If you only need a posterior mean, a few hundred is plenty; if you're reporting tail quantiles or extreme-event probabilities, you want tail-ESS comfortably into the thousands. **Low ESS is not "wrong" the way bad $\hat R$ is — it means imprecise, not biased — but it's a strong smell that the geometry is hard, and the fixes (non-centering, higher `target_accept`) are usually the same.**

### MCSE — how many digits of your answer are real

ESS feeds directly into the **Monte Carlo standard error**, the sampling error of a posterior *estimate* due to using finitely many draws:

$$
\mathrm{MCSE}_{\text{mean}} \;\approx\; \frac{\mathrm{sd}}{\sqrt{\mathrm{ESS}}}.
$$

This is the most underrated column in `az.summary`. It answers the question "how precisely do I actually know this posterior mean?"

> 🧮 **Worked numbers:** Suppose `az.summary` reports for some parameter `mean = 2.43`, `sd = 1.10`, `ess_bulk = 480`. Then $\mathrm{MCSE}_{\text{mean}} \approx 1.10/\sqrt{480} \approx 0.050$. So your estimate of the mean is $2.43 \pm 0.05$ *from Monte Carlo error alone*. The "2.4" is solid; the "3" in the hundredths place is **noise** — running again with a different seed could give 2.38 or 2.47. Reporting "the posterior mean is 2.43" to three significant figures is false precision. If you need that third digit to be real, you need to **quadruple your ESS** (because MCSE shrinks like $1/\sqrt{\mathrm{ESS}}$).

> 🔧 **In practice:** `az.summary(idata)` gives you `mcse_mean` and `mcse_sd` columns for free. A good habit: when you write a posterior mean into a report or a slide, **round it to its MCSE.** Do not show digits the Monte Carlo error has erased. This single discipline will make you look — and be — far more careful than most practitioners.

The three numbers — $\hat R$, ESS, MCSE — are computed together. Let's see the function that produces them.

```python
az.summary(idata, var_names=["mu", "tau"])
# columns: mean  sd  hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
```

You can also call them standalone, which is handy in tests/assertions:

```python
az.rhat(idata, method="rank")          # method="rank" is the Vehtari 2021 default
az.ess(idata, method="bulk")           # also "tail", "mean", "median", "quantile"
az.mcse(idata, method="mean")          # also "sd", "quantile"
```

---

## 5. Visual diagnostics I: trace plots and the "fuzzy caterpillar"

Numbers are summaries; plots show you the *shape* of the failure. The most famous is the **trace plot**, produced by `az.plot_trace(idata)`. For each parameter it draws two panels:

- **Left panel:** the marginal posterior **density (KDE)**, drawn **once per chain** and overlaid. If the chains agree, the per-chain KDEs sit on top of each other.
- **Right panel:** the **trace** — the sampled value plotted against iteration number, **one line per chain**, overlaid.

```python
az.plot_trace(idata, var_names=["mu", "tau"])
```

> 🩺 **Diagnostic — reading a trace plot:** A **healthy** trace looks like a **"fuzzy caterpillar"**: a dense, stationary, horizontal band of noise with no trend, no drift, and no slow wandering. All four chains are tangled together and indistinguishable — you can't tell where one chain ends and another begins. The left-panel KDEs lie on top of each other.
>
> An **unhealthy** trace shows one or more of:
> - **Drift / trend** — a chain that slopes up or down instead of staying level: it hasn't reached stationarity.
> - **A stuck chain** — a flat line, or a chain exploring a clearly different band than the others: a sign of multimodality or a chain trapped. The left-panel KDE for that chain will be visibly displaced.
> - **Slow, snaking excursions** — broad, slow wanders rather than fast fuzz: high autocorrelation, low ESS.
> - **Spikes** — sudden flat sections (the sampler stuck, repeatedly rejecting).

ASCII of the two regimes:

```
GOOD (fuzzy caterpillar)              BAD (drift + a stuck chain)
value                                 value
 |  /\/\/\/\/\/\/\/\/\/\/\/\           |        ________________   <- chain stuck high
 | \/\/\/\/\/\/\/\/\/\/\/\/\          |   ___/
 |  /\/\/\/\/\/\/\/\/\/\/\/\          |  /        /\/\/\/\/\/\    <- others drifting up
 +---------------------------- iter    +----------------------------- iter
```

The trace plot is intuitive and you should always look at it. But the people who invented modern $\hat R$ have a complaint about it, which leads to the better tool.

---

## 6. Visual diagnostics II: rank plots (the modern replacement) and forest plots

### Why rank plots beat trace plots

Trace plots have a real weakness: with four chains and a few thousand draws each, they become a tangle of overlapping ink. Subtle problems — one chain spending slightly more time at high values than the others — are easy to miss in the visual noise. The same paper that gave us rank-normalized $\hat R$ (Vehtari et al. 2021) recommends a more sensitive visual: the **rank plot**.

Here's the idea. Pool *all* the draws across all chains and replace each draw by its **rank** in that pooled sample. If every chain is sampling the same distribution, then within each chain the ranks should be **uniformly distributed** — each chain should contain its fair share of low, medium, and high ranks. A chain that sits systematically higher than the others will be over-represented in the high ranks and under-represented in the low ranks, and its rank histogram will be visibly skewed.

```python
az.plot_rank(idata, var_names=["mu", "tau"], kind="bars")   # kind ∈ {"bars","vlines"}
```

- `kind="bars"` (default) draws **one rank histogram per chain**. **You want every histogram to look flat/uniform.**
- `kind="vlines"` draws each chain's deviation from uniformity as vertical lines from a baseline — small lines = good.

> 🩺 **Diagnostic — reading a rank plot:** **Uniform (flat) bars for every chain = good mixing**, the chains are interchangeable. A chain whose bars **slope up** (too many high ranks) is sitting at systematically higher values than its peers; one whose bars **slope down** sits lower. A **U-shape or hump** means a chain over-explores the extremes or the center. The rank plot makes these location/scale disagreements pop visually in a way the tangled trace plot hides — which is exactly why it's the recommended modern default.

ASCII:

```
GOOD rank histogram (one per chain)     BAD (this chain skews high)
count                                   count
 |  ▇ ▇ ▇ ▇ ▇ ▇ ▇ ▇                      |              ▇ ▇
 |  ▇ ▇ ▇ ▇ ▇ ▇ ▇ ▇   (flat)            |        ▇ ▇ ▇ ▇ ▇   (sloping up)
 |  ▇ ▇ ▇ ▇ ▇ ▇ ▇ ▇                      |  ▇ ▇ ▇ ▇ ▇ ▇ ▇ ▇
 +------------------- rank               +------------------- rank
   low          high                       low          high
```

> 🔧 **In practice:** Use **both**, but trust the rank plot more for convergence. The trace plot is great for *recognizing the kind* of problem (drift vs stuck vs sticky); the rank plot is better for *detecting* a subtle problem at all. In a code review, "show me the rank plots" is the more rigorous ask.

### Forest plots — comparing many parameters at once

When you have a vector of related parameters — the eight school effects $\theta_1,\dots,\theta_8$, or the varying intercepts of a hierarchical model in Chapter 10 — you don't want eight separate posterior plots. The **forest plot** stacks their credible intervals on one axis:

```python
az.plot_forest(idata, var_names=["theta"], combined=True, r_hat=True, ess=True)
```

Each row is one parameter, drawn as two nested segments around a dot (the point estimate): the **thick** line is the inner **50% interval** and the **thin** line the outer **94% HDI** (both controllable via `hdi_prob=`). With `combined=True` the chains are pooled into one interval per parameter (set `combined=False` to see one interval *per chain* — a quick visual convergence check: the per-chain intervals should overlap heavily). The flags `r_hat=True` and `ess=True` print those diagnostics next to each row, so a forest plot doubles as a compact diagnostic dashboard.

> 💡 **Intuition:** The forest plot is the picture of **shrinkage** you'll lean on heavily in Chapter 10 (Hierarchical Models). When you see the eight school effects pulled in toward a common mean, with the noisiest schools pulled in the most, the forest plot is where that story becomes visible. For now, treat it as the natural way to eyeball a whole vector of parameters and their convergence at a glance.

---

## 7. Divergences: HMC's truth-tellers

Now we reach the diagnostic that is unique to Hamiltonian samplers and is, in practice, the one that saves you the most often.

### The geometry: curvature too high for the step size

Recall from Chapter 05 how NUTS moves: it simulates a physical trajectory through parameter space by **leapfrog integration** with a **fixed step size** $\epsilon$, chosen during warmup. Leapfrog is a wonderful integrator — symplectic, volume-preserving, time-reversible — but it is still a *discrete approximation* to a continuous trajectory, with error that grows with the local **curvature** of the log-posterior relative to $\epsilon$.

When the posterior has a region of **extreme curvature** — a place where the geometry bends sharply on a scale smaller than $\epsilon$ — the leapfrog integrator can't keep up. The simulated trajectory, instead of gliding along a level set of the Hamiltonian, shoots off to absurd energy values. The total energy $H$, which exact dynamics would conserve, instead **blows up**. When PyMC detects that the energy error has exceeded a threshold, it flags that trajectory as a **divergence** and abandons it.

> 🧮 **The math (why curvature breaks a fixed step):** Leapfrog error per step is $O(\epsilon^2)$ *for smooth regions*, but the constant hidden in that $O(\cdot)$ scales with the local curvature of $\log p$. In a narrow neck where the scale of variation collapses toward zero, that constant explodes, so for any fixed $\epsilon$ the integrator becomes unstable. Shrinking $\epsilon$ (which raising `target_accept` does) helps — up to a point — but if the curvature varies by orders of magnitude across the posterior, **no single $\epsilon$ works everywhere**: small enough for the neck is wastefully tiny everywhere else. That impossibility is the real reason the cure is to **change the geometry** (reparameterize), not just shrink the step.

> 📜 **Citation/Origin:** Betancourt's phrase is that divergences are **"truth-tellers."** A divergence is not a nuisance warning to be silenced — it is the sampler honestly reporting "there is a region here I could not integrate, so I am systematically failing to explore it, so your posterior is biased." A random-walk Metropolis sampler in the same situation would give you no warning at all — it would just silently under-sample the hard region and report a tidy, wrong posterior. **This is the central reason we prefer gradient-based samplers: they tell you when they fail.**

### Why even a few divergences matter

> ⚠️ **Pitfall:** "It's only 12 divergences out of 4000 draws, surely that's fine?" **No.** Divergences are not random noise spread uniformly over the posterior — they **cluster** in the exact region the sampler can't handle (usually the funnel neck, §8). So even a handful of divergences signals that an entire region of parameter space is being systematically under-explored, which biases every expectation that depends on that region. The target is **zero divergences.** A small number after a fix is a "look harder," not a "ship it."

The first line of your loop, again:

```python
n_div = int(idata.sample_stats["diverging"].sum())
print(f"{n_div} divergences")          # the target is 0
```

### Localizing divergences with `az.plot_pair`

Knowing you have divergences is step one. Knowing **where** they are in parameter space is what tells you the *cause* and points you at the fix. The tool is a pair plot (a 2-D scatter of two parameters) with the divergent draws highlighted:

```python
az.plot_pair(
    idata,
    var_names=["theta", "tau"],
    coords={"theta_dim_0": 0},   # pick one component of the theta vector
    divergences=True,            # highlight divergent transitions
)
# NOTE on "theta_dim_0": that is the dimension name PyMC auto-generates when you declare a
# vector parameter with shape=J. If instead you build the model with pm.Model(coords=...)
# and dims="school" (as we do in the worked example in §9), the selector reads coords={"school": 0}.
# NOTE: newer ArviZ (arviz-plots v1.x) uses visuals={"divergence": True} instead.
# This course is frozen on the legacy az.* API that PyMC v5 returns: use divergences=True.
```

> 🩺 **Diagnostic — reading a divergence pair plot:** The non-divergent draws form the cloud of the joint posterior; the divergent draws are over-plotted (typically in a contrasting color). **Look at where the divergent points pile up.** If they cluster in one corner — characteristically where a group-scale parameter like $\tau$ gets **small** — you've found a **funnel** (next section), and the cure is non-centering. If they're scattered everywhere with no pattern, suspect a step size that's just slightly too large (raise `target_accept`) or too little tuning. The *spatial pattern of divergences is the diagnosis.*

> 🔧 **In practice:** Funnels live in a region of **small** scale, and small positive numbers are bunched up against zero on a linear axis — so the funnel neck is squashed and invisible. **Plot the scale parameter on a log axis.** The standard trick (straight from the PyMC divergences gallery) is to add `pm.Deterministic("log_tau", pm.math.log(tau))` to the model and pair `theta` against `log_tau`. On the log axis the funnel opens up into its characteristic trumpet shape and the divergences line up neatly in the neck.

---

## 8. The funnel, and the energy diagnostic

### Neal's funnel: the archetypal pathology

Almost every divergence you meet in hierarchical modeling is a **funnel**, so let's understand it in its purest form. Radford Neal's funnel is a tiny two-line model engineered to be hard:

$$
v \sim \mathrm{Normal}(0,\,3), \qquad x_i \mid v \sim \mathrm{Normal}\!\left(0,\; e^{v/2}\right),\quad i=1,\dots,9.
$$

```python
with pm.Model() as funnel:
    v = pm.Normal("v", 0.0, 3.0)
    x = pm.Normal("x", 0.0, pm.math.exp(v / 2), shape=9)
    idata_funnel = pm.sample(1000, tune=1000, chains=4,
                             target_accept=0.95, random_seed=RANDOM_SEED)
```

Stare at the geometry. The width of $x$ is controlled by $v$: when $v$ is large (say $+6$), $x$ has standard deviation $e^3 \approx 20$ — a wide, gentle mouth. When $v$ is small (say $-6$), $x$ has standard deviation $e^{-3} \approx 0.05$ — a razor-thin neck. The joint density $(v, x)$ looks like a **funnel** (or a trumpet): wide at the top, pinching to nothing at the bottom. The curvature in the neck is extreme: a step size $\epsilon$ tuned for the wide mouth is catastrophically too large for the neck, so the sampler **diverges** as it tries to descend into the neck — and the neck is precisely where a lot of the probability mass lives.

> 💡 **Intuition:** This is not a contrived special case — it is the *shape of every hierarchical model*. Whenever a group of parameters shares a common scale ($\theta_j \sim \mathrm{Normal}(\mu, \tau)$), the $\theta_j$'s width is governed by $\tau$, and small $\tau$ squeezes them into a neck. The Eight Schools model in §9 is a funnel in disguise, and so is every varying-intercept model in Chapter 10. **Master the funnel and you've mastered the recurring villain of applied Bayesian computation.**

### The energy plot and BFMI

There is a second, subtler pathology that divergences don't always catch, and it has its own diagnostic: the **energy plot**.

Recall NUTS alternates two operations: it resamples the momentum (which changes the total energy $H$), then it flows along a trajectory at roughly fixed energy. For the sampler to explore the full posterior, the momentum resampling step must be able to move the chain across the **whole range of energies** the posterior spans. If the posterior has heavy tails, it spans a wide range of energies, and a single momentum resample might only nudge the energy a little — so the chain explores the energy range slowly, getting stuck at moderate energies and rarely reaching the tails.

`az.plot_energy(idata)` visualizes this by overlaying two densities:

- $\pi_E$ — the **marginal energy distribution** (the spread of energies the chain visited).
- $\pi_{\Delta E}$ — the distribution of **energy transitions** (how much energy changed between successive draws, driven by momentum resampling).

```python
az.plot_energy(idata)
az.bfmi(idata)        # one BFMI value per chain
```

> 🩺 **Diagnostic — reading the energy plot:** You want the two densities to be **similar in width and well-overlapping.** That means each momentum resample can move the chain across a meaningful chunk of the energy range — the chain can reach the tails. If the **transition** density ($\pi_{\Delta E}$) is **much narrower** than the **marginal** density ($\pi_E$), the resampling can't traverse the energy range; the chain explores the tails slowly and your tail estimates (and tail-ESS) suffer.

This is quantified by the **E-BFMI**, the energy Bayesian Fraction of Missing Information:

> 🧮 **The math:**
> $$
> \text{E-BFMI} \;=\; \frac{\mathbb{E}\big[(E_n - E_{n-1})^2\big]}{\mathrm{Var}(E_n)},
> $$
> the mean squared energy *transition* over the marginal energy *variance*. Small E-BFMI ⇒ transitions are tiny relative to the spread ⇒ slow energy exploration.

> 🩺 **Diagnostic — the BFMI rule:** **BFMI $> 0.3$ is healthy; BFMI $< 0.3$ flags an energy problem** — typically heavy tails the momentum can't traverse, often cured by reparameterizing or using a lighter-tailed / tighter prior on a scale parameter. **Heads-up on the number:** Stan's older `check_energy` warned below roughly **0.2**, so you'll see both 0.2 and 0.3 in the wild. This course standardizes on **0.3** (consistent with the threshold table in the appendix); just know that a value of, say, 0.25 is borderline and worth investigating.

> ⚠️ **Pitfall:** A funnel can produce a low BFMI *and* divergences together — they're two views of the same heavy-tailed, high-curvature geometry. Don't treat them as separate bugs to fix separately; they usually share one cure: **non-centering** (next).

---

## 9. The big one: centered vs non-centered, worked end-to-end on Eight Schools

This is the section that justifies the chapter. We will fit the most famous hierarchical model in statistics **twice** — once the naive way (centered), where it breaks, and once reparameterized (non-centered), where it sings — and narrate **every diagnostic** as it goes from red to green. Internalize this and you can debug 80% of the hierarchical-model failures you'll ever meet.

### The data and the model

The **Eight Schools** dataset (Rubin, 1981; the canonical hierarchical example, in our dataset catalog) reports the estimated effect of an SAT-coaching program at eight schools, each with its own measurement standard error:

```python
J = 8
y     = np.array([28.,  8., -3.,  7., -1.,  1., 18., 12.])   # estimated effects
sigma = np.array([15., 10., 16., 11.,  9., 11., 10., 18.])   # known standard errors
```

The model is the textbook partial-pooling hierarchy: each school has a true effect $\theta_j$, the $\theta_j$ are drawn from a common population $\mathrm{Normal}(\mu, \tau)$, and we observe each $\theta_j$ with known noise $\sigma_j$:

$$
\mu \sim \mathrm{Normal}(0, 5), \qquad
\tau \sim \mathrm{HalfCauchy}(5), \qquad
\theta_j \sim \mathrm{Normal}(\mu, \tau), \qquad
y_j \sim \mathrm{Normal}(\theta_j, \sigma_j).
$$

Here $\mu$ is the overall coaching effect, $\tau$ is how much schools genuinely differ, and the magic of partial pooling is that small-$\tau$ posteriors shrink the noisy school estimates toward $\mu$. The trouble is that **$\tau$ small is exactly the funnel neck.**

> 🔧 **In practice — about that `HalfCauchy(5)` on $\tau$:** this is the *historical* Eight Schools choice (Gelman's 2006 prior-on-variance paper), and we keep it so our numbers match the canonical gallery. But notice the tension with §8's BFMI advice: a `HalfCauchy` is itself **heavy-tailed**, and a scale parameter free to wander far out into its tail is precisely what stretches the energy range and stresses BFMI. A `HalfNormal(5)` or `Exponential` would be **lighter-tailed** and is often preferred today (see Chapters 02–03 on choosing scale priors); it tames the *tail* of $\tau$. What it does **not** fix is the funnel *geometry* — that comes from the multiplicative coupling $\theta_j \sim \mathrm{Normal}(\mu, \tau)$ and survives any choice of prior. So the two cures address two different ailments: a lighter prior helps the energy/BFMI tail problem, while **non-centering** (next) is what tames the small-$\tau$ curvature regardless of the prior's tail. Here we deliberately keep the heavy-tailed prior so the funnel shows up in full force.

> 🔧 **In practice — the $\sigma$ convention, said once and for all:** PyMC and Stan parameterize the Normal by its **standard deviation** $\sigma$, not the variance and *definitely* not the precision $1/\sigma^2$. If you learned Bayesian modeling in BUGS or JAGS, burn this in: those tools used **precision**, and silently feeding a "precision-sized" number into PyMC's `sigma` is a classic, catastrophic footgun. In this course, `sigma=` always means standard deviation.

### Attempt 1 — the centered parameterization (this is the bug)

```python
coords = {"school": np.arange(J)}          # named dim -> readable output & selectors

with pm.Model(coords=coords) as centered:
    mu  = pm.Normal("mu", mu=0.0, sigma=5.0)
    tau = pm.HalfCauchy("tau", beta=5.0)
    log_tau = pm.Deterministic("log_tau", pm.math.log(tau))      # <-- so the funnel neck is visible on a log axis
    theta = pm.Normal("theta", mu=mu, sigma=tau, dims="school")  # <-- centered: theta tied to (mu, tau)
    obs   = pm.Normal("obs", mu=theta, sigma=sigma, observed=y)

    idata_c = pm.sample(1000, tune=1000, chains=4,
                        target_accept=0.9, random_seed=RANDOM_SEED)
```

> 🔧 **In practice — named dims, not bare `shape`:** because we passed `coords={"school": ...}` to `pm.Model` and wrote `dims="school"`, the `theta` dimension is now called `school` in every plot, summary, and selector. Had we written `shape=J` instead, PyMC would have auto-named the dimension `theta_dim_0` — which is where that magic string comes from when you see it in other people's `coords=` calls. Named dims make the `coords=` selector below read as plain English (`{"school": 0}`) instead of a positional incantation. We defined `log_tau` here, *inside* the model graph, because a `Deterministic` must exist **before** `pm.sample` to be recorded in `idata_c` — you cannot bolt one on after the fact.

The defining line is `theta = pm.Normal("theta", mu=mu, sigma=tau, dims="school")`. The prior **width** of $\theta$ is $\tau$, a parameter we are also estimating. That coupling is the whole problem: when the sampler proposes a small $\tau$, the $\theta_j$ are squeezed into a sliver, the curvature explodes, and we're in Neal's funnel.

Run it and PyMC will print something like:

```
There were 87 divergences after tuning. Increase `target_accept` or reparameterize.
```

Now read the diagnostics, top to bottom, the way you always will.

**(a) Count divergences:**
```python
print(int(idata_c.sample_stats["diverging"].sum()))     # e.g. 87  -> NOT zero. Stop.
```
Nonzero. The geometry defeated the integrator somewhere. We already know to suspect a funnel.

**(b) `az.summary` — $\hat R$ and ESS:**
```python
az.summary(idata_c, var_names=["mu", "tau"])
```
You'll see something like (numbers vary by seed):

| | mean | sd | hdi_3% | hdi_97% | mcse_mean | mcse_sd | ess_bulk | ess_tail | r_hat |
|---|---|---|---|---|---|---|---|---|---|
| mu | 4.3 | 3.3 | -1.9 | 10.5 | 0.16 | 0.11 | 430 | 690 | **1.02** |
| tau | 3.6 | 3.1 | 0.0 | 9.1 | 0.30 | 0.22 | **95** | **120** | **1.05** |

Read it: `tau` has **$\hat R = 1.05$** (above 1.01 — chains disagree) and **ESS in the dozens** (below 400 — barely any independent information). The estimate of $\tau$ is both **biased** (the divergences mean the small-$\tau$ neck is under-explored, so the posterior of $\tau$ is pushed *away* from zero) and **imprecise**. This is a textbook fail on every count.

**(c) Trace / rank plots:**
```python
az.plot_trace(idata_c, var_names=["tau"])
az.plot_rank(idata_c, var_names=["tau"])
```
The trace of `tau` shows chains that don't fully overlap and occasional sticky flat sections near zero; the rank plot for `tau` shows clearly **non-uniform** bars — some chains over-represented at the low end. Both confirm the chains are not interchangeable.

**(d) Localize the divergences — see the funnel:**

Because we already defined `log_tau` *inside* the `centered` model above, it lives in `idata_c.posterior` and we can pair against it directly — no after-the-fact surgery needed. We plot `theta` (one component, selected by the named `school` dim) against `log_tau` so the neck opens up on a log axis:

```python
az.plot_pair(
    idata_c,
    var_names=["theta", "log_tau"],
    coords={"school": 0},     # one school, by name (not the auto "theta_dim_0")
    divergences=True,         # newer ArviZ: visuals={"divergence": True}
)
```
The pair plot of $\theta_0$ against $\log\tau$ shows the classic shape: the cloud narrows to a point as $\tau \to 0$ (i.e. $\log\tau \to -\infty$), and **the divergent draws are piled up in that neck.** That picture *is* the diagnosis. (On a linear $\tau$ axis the small-$\tau$ neck is squashed against zero and the divergences hide; the log axis is what stretches it open and makes the funnel unmistakable.)

**(e) Energy:**
```python
az.plot_energy(idata_c)
az.bfmi(idata_c)            # likely some chains < 0.3
```
The energy transition density is noticeably narrower than the marginal; BFMI dips toward or below 0.3 on some chains. Same heavy-tailed-neck story, different view.

Every diagnostic agrees: **the centered model is broken at small $\tau$.** Now the cure.

### Attempt 2 — the non-centered parameterization (this is the fix)

The fix does not change the model — the posterior $p(\mu, \tau, \theta \mid y)$ is mathematically *identical*. It changes the **coordinates** the sampler works in, so that the geometry it has to integrate is nice. Instead of drawing $\theta_j$ directly from $\mathrm{Normal}(\mu, \tau)$ — which welds $\theta_j$'s width to $\tau$ — we draw a **standardized** offset $\tilde\theta_j \sim \mathrm{Normal}(0,1)$ on a fixed, well-conditioned geometry, and then **recover** $\theta_j$ deterministically:

$$
\tilde\theta_j \sim \mathrm{Normal}(0, 1), \qquad \theta_j \;=\; \mu + \tau\,\tilde\theta_j.
$$

> 🧮 **The math (why this is the same model):** If $\tilde\theta_j \sim \mathrm{Normal}(0,1)$, then the affine transform $\theta_j = \mu + \tau\,\tilde\theta_j$ is distributed exactly $\mathrm{Normal}(\mu, \tau)$ — that's the location–scale property of the Normal. So the implied prior on $\theta_j$ is unchanged; only the parameterization the sampler explores is different. Crucially, in the **non-centered** coordinates the parameter the sampler actually moves, $\tilde\theta_j$, has a **fixed unit-scale geometry that does not depend on $\tau$ at all.** The funnel is gone because the coupling that created it is gone. We pay for it only with one cheap deterministic multiply.

```python
with pm.Model(coords=coords) as noncentered:                    # reuse coords={"school": ...}
    mu  = pm.Normal("mu", mu=0.0, sigma=5.0)
    tau = pm.HalfCauchy("tau", beta=5.0)
    log_tau = pm.Deterministic("log_tau", pm.math.log(tau))      # keep it for an apples-to-apples pair plot

    theta_t = pm.Normal("theta_t", mu=0.0, sigma=1.0, dims="school")  # <-- standardized, geometry fixed
    theta   = pm.Deterministic("theta", mu + tau * theta_t, dims="school")  # <-- affine recover, same model

    obs = pm.Normal("obs", mu=theta, sigma=sigma, observed=y)

    idata_nc = pm.sample(1000, tune=1000, chains=4,
                         target_accept=0.9, random_seed=RANDOM_SEED)
```

Now re-run the **same** diagnostic loop and watch every light turn green.

**(a) Divergences:**
```python
print(int(idata_nc.sample_stats["diverging"].sum()))    # 0 (or a tiny handful)
```
Zero, or close to it. The integrator can resolve the geometry everywhere now.

**(b) `az.summary`:**
```python
az.summary(idata_nc, var_names=["mu", "tau"])
```

| | mean | sd | hdi_3% | hdi_97% | mcse_mean | mcse_sd | ess_bulk | ess_tail | r_hat |
|---|---|---|---|---|---|---|---|---|---|
| mu | 4.4 | 3.3 | -1.7 | 10.6 | 0.05 | 0.04 | **3900** | **2900** | **1.00** |
| tau | 3.6 | 3.2 | 0.0 | 9.4 | 0.06 | 0.05 | **2600** | **1900** | **1.00** |

$\hat R = 1.00$ everywhere, **ESS leaps from ~95 into the thousands**, and MCSE shrinks by roughly $\sqrt{\text{ESS ratio}}$ — about a fivefold improvement in precision *for free*, just by changing coordinates. This is the dramatic before/after that makes the technique unforgettable.

**(c) Trace / rank:** the trace of `tau` is now a clean fuzzy caterpillar with all chains tangled; the rank plot bars are flat and uniform.

**(d) Pair plot:**
```python
az.plot_pair(idata_nc, var_names=["theta_t", "log_tau"],
             coords={"school": 0}, divergences=True)   # named dim, same as the centered plot
```
The neck is gone: in the non-centered coordinates the cloud is well-conditioned and there are no divergences to localize.

**(e) Energy:** the marginal and transition energy densities now overlap nicely and BFMI sits comfortably above 0.3.

> 💡 **Intuition — what just happened:** We didn't change *what* we're inferring, we changed *how the sampler crawls over it*. The centered coordinates are like a topographic map drawn on a sheet that's been pinched to a point — the sampler keeps falling into the crease. The non-centered coordinates flatten that sheet back out, so the same landscape is now easy to walk. **Reparameterization is the art of choosing coordinates in which the posterior geometry is benign.** It is the most leveraged debugging move in the toolbox.

> ⚠️ **Pitfall — non-centered is not *always* better.** The non-centered form is the cure in the **weak-data / small-$\tau$** regime — exactly the hierarchical-with-few-observations-per-group case you'll usually be in. But if you have **lots of informative data per group**, the data pins each $\theta_j$ down tightly, the funnel doesn't form, and the **centered** form can actually mix *better* (the non-centered coordinates then become the awkward ones). The general principle: non-center the groups that are data-starved. In Chapter 10 we'll discuss per-group choices; for now, reach for non-centering first when you see a funnel, and remember it's a tool, not a law.

> 📜 **Citation/Origin:** The centered/non-centered story and the Eight Schools funnel come straight from the PyMC example-gallery notebook *"Diagnosing Biased Inference with Divergences"* and from Betancourt's hierarchical-modeling case study. We introduced the move conceptually in Chapter 05; this is its proper home. It returns as the **default cure for funnels** in Chapter 10 (Hierarchical & Multilevel Models) — same terminology, same code shape.

### Closing the loop on the worked example: a posterior predictive check

Diagnostics tell you the sampler did its job; a **posterior predictive check** (PPC) tells you whether the *model* is any good — does data simulated from the fitted model look like the data you actually saw? It's a different question from convergence, and we cover it fully in the Bayesian-workflow chapter (Chapter 08) and use it for model comparison in Chapter 11, but it belongs at the end of every fit, so let's run one here to complete the worked example honestly.

```python
with noncentered:
    # draw replicated datasets from the posterior predictive
    pm.sample_posterior_predictive(idata_nc, extend_inferencedata=True,
                                   random_seed=RANDOM_SEED)

az.plot_ppc(idata_nc, num_pp_samples=100)     # kind="kde" by default
```

> 🩺 **Diagnostic — reading `az.plot_ppc`:** It overlays many light predictive densities (one per posterior-predictive draw) against the **observed** data density (a bold line). You want the observed line to sit comfortably **within the spread** of the predictive densities — the model can reproduce data like what you saw. If the observed line falls outside the predictive cloud (e.g., the model never predicts the spread or the extremes you actually observe), the *model* is misspecified, even if every convergence diagnostic was perfect. With only eight points the Eight Schools PPC is coarse, but it should show the observed effects landing inside the predictive envelope. (For sharper, double-dipping-free calibration checks — LOO-PIT and Bayesian $p$-values — see Chapters 08 and 11; they need a `log_likelihood` group, which you can add with `pm.compute_log_likelihood(idata_nc, model=noncentered)` if your PyMC version didn't compute it during sampling.)

> ⚠️ **Pitfall:** `az.plot_ppc` will raise "no `posterior_predictive` group" if you forget to call `pm.sample_posterior_predictive(...)` first. The PPC group is not produced by `pm.sample`; you have to ask for it.

---

## 10. The tuning recipe: a deliberate order of operations

When a model throws divergences, low ESS, or bad $\hat R$, you do **not** flail. There is a recipe, and you escalate through it in order, because the cheap fixes are also the ones that teach you the most about *why* the model is hard. Here is the order I want you to follow.

**Step 0 — Read the diagnostics and localize the failure first.** Never tune blind. Count divergences, run `az.summary`, and `az.plot_pair(..., divergences=True)`. If the divergences cluster in a funnel neck, jump straight to reparameterizing — no amount of `target_accept` fully cures a funnel.

**Step 1 — Raise `target_accept`: $0.9 \to 0.95 \to 0.99$.** This tells NUTS's dual-averaging step-size adaptation to aim for a higher acceptance rate, which forces a **smaller** step size $\epsilon$. Smaller steps resolve higher curvature, so borderline divergences often vanish. It's the first thing to try because it's a one-character change.

```python
idata = pm.sample(1000, tune=1000, chains=4,
                  target_accept=0.95, random_seed=RANDOM_SEED)   # then 0.99 if needed
```

> 🔧 **In practice:** Higher `target_accept` is **not free** — smaller steps mean longer trajectories, deeper trees, slower sampling, and **diminishing returns past ~0.99.** If you find yourself at `target_accept=0.99` and *still* diverging, the message is not "go to 0.999" — it's "the geometry is wrong; reparameterize." `target_accept` buys you out of *mild* curvature, not a funnel.

**Step 2 — Increase `tune`.** Warmup is when NUTS adapts both the step size and the mass matrix. If `tune` is too short, those are mis-set and you get *spurious* divergences and low ESS that more tuning alone fixes. For hard models, bump `tune` from 1000 to **2000–3000**.

```python
idata = pm.sample(1000, tune=3000, chains=4,
                  target_accept=0.95, random_seed=RANDOM_SEED)
```

**Step 3 — Adapt a full mass matrix when parameters are correlated.** The default `init="jitter+adapt_diag"` learns only a **diagonal** mass matrix — per-parameter scales, but no cross-correlations. If your posterior has strong correlations between parameters (a tilted, cigar-shaped joint), a diagonal metric can't capture the tilt, and you'll see **slow sampling and low ESS with *no* divergences** — the tell-tale signature. The cure is a **dense** mass matrix:

```python
idata = pm.sample(1000, tune=2000, chains=4, target_accept=0.95,
                  init="jitter+adapt_full",          # learn the full covariance metric
                  random_seed=RANDOM_SEED)
```

`adapt_full` costs $O(d^2)$ per step, so use it when correlations are the problem, not by default. (Often the *better* fix is to reparameterize away the correlation — but `adapt_full` is the quick win.)

**Step 4 — Reparameterize (the real fix for funnels).** If diagnostics point at a funnel — divergences in a neck, low BFMI, low ESS on a scale parameter — **non-center** the hierarchy (§9). This is not a knob you turn; it's a change of coordinates, and it's the cure no amount of `target_accept` or `tune` substitutes for. For non-funnel pathologies, "reparameterize" might mean standardizing predictors, log-transforming a positive parameter, or decorrelating with a QR/Cholesky trick. The principle is constant: **choose coordinates in which the posterior geometry is benign.**

**Step 5 — Only now, raise `max_treedepth`.** If you see many "maximum tree depth reached" warnings, NUTS is hitting its cap on trajectory length (default `max_treedepth=10`, so up to $2^{10}=1024$ leapfrog steps). This is an **efficiency** warning, not a bias warning — your draws aren't wrong, just expensive. **Resist the urge to simply raise the cap.** Hitting max treedepth is almost always a *symptom* of a poorly scaled metric or a too-small step size, both upstream problems. Fix the geometry first (Steps 1–4); only raise `max_treedepth` if you've genuinely got a long-but-healthy posterior.

```python
idata = pm.sample(1000, tune=2000, nuts={"max_treedepth": 12},
                  target_accept=0.95, random_seed=RANDOM_SEED)
```

> 💡 **Intuition — the mental model for the whole recipe:** divergences and "max treedepth reached" usually mean the **step size** is being forced small by bad geometry; low ESS with no divergences usually means **correlation** the diagonal metric misses; bad $\hat R$ means **chains in different places**. `target_accept` and `tune` adjust the integrator's *settings*; `adapt_full` and reparameterization fix the *geometry the integrator sees*. Always prefer fixing the geometry — that's the lesson the funnel keeps teaching.

> ⚠️ **Pitfall:** Do **not** "fix" divergences by silencing the warning or by lowering `target_accept` until the warning stops appearing — that just hides the bias. Divergences are information; the only legitimate responses are to remove their *cause* (reparameterize, tighten priors, rescale) or to shrink the step (raise `target_accept`) until they genuinely stop.

---

## 11. The complete diagnostic loop (your checklist)

Here is the loop, in order, that you run on **every** fit before you interpret a single posterior number. Memorize the shape of it; it's the spine of being a trustworthy Bayesian.

```python
# 0. SAMPLE — always multiple chains, jittered starts, a sane target_accept.
idata = pm.sample(1000, tune=1000, chains=4,
                  target_accept=0.9, random_seed=RANDOM_SEED)

# 1. DIVERGENCES — the hard gate. Must be 0.
print("divergences:", int(idata.sample_stats["diverging"].sum()))

# 2. R-HAT & ESS — scan the table. r_hat <= 1.01 ; ess_bulk & ess_tail >= 400.
print(az.summary(idata))

# 3. RANK PLOTS (preferred) + trace plots — chains interchangeable? fuzzy caterpillars?
az.plot_rank(idata)
az.plot_trace(idata)

# 4. ENERGY & BFMI — densities overlap? bfmi > 0.3?
az.plot_energy(idata)
print("bfmi:", az.bfmi(idata))

# 5. LOCALIZE any divergences — where do they cluster? (funnel neck => non-center)
# Pick 2-3 params, especially any scale param + a group effect; passing no var_names
# renders the full all-pairs grid (dozens of subplots), which is slow and unreadable.
az.plot_pair(idata, var_names=["theta", "log_tau"], coords={"school": 0}, divergences=True)

# 6. POSTERIOR PREDICTIVE CHECK — is the MODEL any good? (Ch 08/11 go deeper)
pm.sample_posterior_predictive(idata, extend_inferencedata=True, random_seed=RANDOM_SEED)
az.plot_ppc(idata)
```

The decision rule is brutally simple. **If step 1 is nonzero, or step 2 shows any $\hat R > 1.01$ or any ESS $< 400$, you stop and debug (§10). You do not report the posterior.** Steps 3–5 tell you *what kind* of problem you have so you can pick the right fix. Step 6 is a different gate — model adequacy, not sampler adequacy — and you'll lean on it harder in Chapters 08 and 11. A model has to clear **both** gates: the sampler must converge **and** the model must fit.

---

## ⚠️ Common errors & how to fix them

The master symptom → cause → fix table. This is the page to keep open while you debug. It builds on the divergence/energy mechanics from Chapter 05 and is the diagnostic-specific companion to the troubleshooting table in the Appendix.

| Symptom (message / number you see) | What it means (cause) | Fix (in order) |
|---|---|---|
| `There were N divergences after tuning` | Step size too large for a high-curvature region (almost always a **funnel neck**); the integrator's energy blew up | 1) `az.plot_pair(..., divergences=True)` to localize; 2) **non-center** if it's a funnel; 3) raise `target_accept` 0.9→0.95→0.99; 4) tighten/regularize the scale prior; 5) standardize predictors |
| `r_hat > 1.01` (e.g. 1.05, 1.3) | Chains haven't converged to a common distribution — too little tuning, pathological geometry, or genuine **multimodality** | More `tune`/`draws`; check `plot_rank`/`plot_trace` for a stuck chain; reparameterize; question model **identifiability** |
| `ess_bulk` or `ess_tail` tiny (e.g. 30–100) | High autocorrelation / poor geometry. *Imprecise, not biased* — but a strong smell | Non-center; raise `target_accept`; longer chains; **rescale/standardize predictors**; if correlated with no divergences → `init="jitter+adapt_full"` |
| Slow sampling, low ESS, **zero divergences** | Strong posterior **correlations** the diagonal mass matrix can't capture (tilted cigar) | `init="jitter+adapt_full"` (dense metric); better, **reparameterize to decorrelate** (e.g. QR, Cholesky, centering choice) |
| `BFMI < 0.3` / energy warning; transition-energy density much narrower than marginal | **Heavy tails**; momentum resampling can't traverse the energy range | Reparameterize; lighter-tailed or tighter prior on the offending scale param; non-center |
| `r_hat = NaN` in the summary | Only **1 chain** (R̂ needs ≥2), or the variable is a constant / fixed `Deterministic` | Sample `chains=4`; ignore for genuinely fixed Deterministics |
| Many `Maximum tree depth reached` / `tree_depth == 10` | Trajectories truncated at the cap — **inefficiency, not bias**; usually a poorly scaled metric or tiny step | Fix geometry first (standardize, `adapt_full`, more `tune`); only then `nuts={"max_treedepth": 12}` |
| `The acceptance probability does not match the target` | Step-size adaptation didn't converge — too few tuning steps | Increase `tune` (2000–3000); keep `init="jitter+adapt_diag"` |
| `SamplingError: Initial evaluation ... results in -inf` | Bad initial point / a value impossible under the prior (e.g. observed outside support) | Set `initvals=`; fix priors/transforms; check the domain of `observed` data |
| `az.plot_ppc` raises "no posterior_predictive group" | You never ran `sample_posterior_predictive` | `pm.sample_posterior_predictive(idata, extend_inferencedata=True)` first |
| `az.loo_pit` / LOO errors on missing log-likelihood | The `log_likelihood` group isn't in the InferenceData | `pm.compute_log_likelihood(idata, model=model)`, or sample with `idata_kwargs={"log_likelihood": True}` |
| `plot_pair(..., divergences=True)` raises `TypeError` | You're on the **new** `arviz-plots` v1.x package, not legacy `arviz` | Use legacy `arviz` (the frozen course stack), or switch to `visuals={"divergence": True}` |
| Divergences only at small `tau`, invisible on a linear axis | A **funnel** squashed against zero | Add `pm.Deterministic("log_tau", pm.math.log(tau))` and pair `theta` vs `log_tau` — the neck opens up on the log scale |
| PPC looks perfect but you suspect the model is wrong | **Double-dipping** (PPC reuses the data) or an uninformative test statistic | Use **LOO-PIT** + informative `T` (min/max/sd/#zeros), not just location stats — Ch 08/11 |

---

## 🧪 Exercises

1. **(Conceptual — $\hat R$ from first principles.)** In your own words, explain why the *split* in "split-$\hat R$" is necessary. Construct (on paper or in a sketch) a single chain that has the *same* overall mean and variance as a converged chain but is clearly non-stationary, and argue why classic (unsplit) $\hat R$ would miss it while split-$\hat R$ catches it. *Hint: think about a chain that spends its first half low and its second half high.*

2. **(Conceptual — bulk vs tail.)** Give a concrete modeling situation where you'd see **high bulk-ESS but low tail-ESS**, and explain which posterior summaries you could trust and which you could not in that case. Then state, in terms of MCSE, how many *more* draws you'd need to halve the Monte Carlo error of a posterior mean. *Hint: MCSE $\propto 1/\sqrt{\mathrm{ESS}}$.*

3. **(Coding — reproduce the funnel.)** Fit Neal's funnel (the 2-line model in §8) with `target_accept=0.9`, count the divergences, then refit at `0.99`. Report how the divergence count changes. Now **non-center** the funnel (`x_t ~ Normal(0,1)`, `x = pm.Deterministic("x", x_t * pm.math.exp(v/2))`) and show divergences drop to ~0 and tail-ESS jumps. *Hint: the cure is structurally identical to Eight Schools — the scale that creates the neck is `exp(v/2)`.*

4. **(Coding — the full loop on Eight Schools.)** Run the complete §11 diagnostic loop on **both** the centered and non-centered Eight Schools models. For each, produce: the divergence count, the `az.summary` for `mu` and `tau`, a rank plot for `tau`, and a `plot_pair` of `theta[0]` vs `log_tau` with divergences highlighted. Write two sentences per diagnostic explaining what you see. *Hint: add `pm.Deterministic("log_tau", pm.math.log(tau))` to both models so the pair plots are comparable.*

5. **(Coding — `target_accept` cannot save a funnel.)** Take the *centered* Eight Schools model and sweep `target_accept` over `[0.8, 0.9, 0.95, 0.99]`, recording the divergence count and `ess_bulk` for `tau` at each. Plot divergences vs `target_accept`. Use the result to argue *why* the §10 recipe says "reparameterize" rather than "just push `target_accept` higher." *Hint: you should see divergences fall but not vanish, and ESS stay poor — the geometry, not the step size, is the bottleneck.*

6. **(Diagnostic literacy — the silent correlation trap.)** Simulate a 2-D posterior with strong correlation (e.g. a regression with two highly collinear predictors, or sample `pm.MvNormal` with off-diagonal covariance near 0.99). Fit with the default diagonal metric, observe **low ESS and zero divergences**, then refit with `init="jitter+adapt_full"`. Report the ESS change. Explain why this pathology produces *no* divergences (so the divergence count alone would have missed it) and what diagnostic *did* flag it. *Hint: this is the one place where reading ESS, not divergences, is what saves you.*

---

## 📚 Resources & further reading

The canonical sources behind every threshold and plot in this chapter. Read at least the first two if you read nothing else.

- **Vehtari, Gelman, Simpson, Carpenter, Bürkner (2021), "Rank-normalization, folding, and localization: An improved $\hat R$ for assessing convergence of MCMC,"** *Bayesian Analysis* 16(2):667–718 — the source of $\hat R \le 1.01$, bulk/tail-ESS, and rank plots. Read §3 ($\hat R$), §4 (ESS), §5 (rank plots). https://projecteuclid.org/journals/bayesian-analysis/volume-16/issue-2/Rank-Normalization-Folding-and-Localization--An-Improved-R%CB%86-for/10.1214/20-BA1221.full
- **Online appendix + worked formulas** (same authors) — copy-pasteable definitions and exact recommended thresholds. https://avehtari.github.io/rhat_ess/rhat_ess.html
- **PyMC example gallery — "Diagnosing Biased Inference with Divergences"** — the single best hands-on reference for this chapter: the exact centered/non-centered Eight Schools code, `sample_stats.diverging`, and the funnel pair plot. https://www.pymc.io/projects/examples/en/latest/diagnostics_and_criticism/Diagnosing_biased_Inference_with_Divergences.html
- **PyMC example gallery — "Sampler Statistics"** — every field in `idata.sample_stats` (energy, tree_depth, step_size, acceptance_rate) and how to access it. https://www.pymc.io/projects/examples/en/latest/diagnostics_and_criticism/sampler-stats.html
- **Betancourt, "A Conceptual Introduction to Hamiltonian Monte Carlo,"** arXiv:1701.02434 — *why* divergences happen, the geometry of the typical set and curvature. §3–5. https://arxiv.org/abs/1701.02434
- **Betancourt, "Robust Statistical Workflow with PyStan"** — the geometric story of divergences and BFMI, and the `check_*` diagnostic functions (Stan idioms, same concepts). https://betanalpha.github.io/assets/case_studies/pystan_workflow.html
- **Gabry, Simpson, Vehtari, Betancourt, Gelman (2019), "Visualization in Bayesian workflow,"** *JRSS-A* 182(2):389–402 — origin of the energy plot and the PPC/LOO-PIT visual idioms ArviZ uses. https://doi.org/10.1111/rssa.12378 (code: https://github.com/jgabry/bayes-vis-paper)
- **ArviZ docs** — `az.summary` (columns, `kind`, `stat_focus`, "r_hat needs ≥2 chains"): https://python.arviz.org/en/v0.21.0/api/generated/arviz.summary.html · `az.plot_rank`: https://python.arviz.org/en/latest/api/generated/arviz.plot_rank.html · `az.plot_energy`: https://python.arviz.org/en/stable/api/generated/arviz.plot_energy.html · `az.bfmi`: https://python.arviz.org/en/latest/api/generated/arviz.bfmi.html
- **ArviZ InferenceData schema spec** — the authoritative meaning of every group (`posterior`, `sample_stats`, `log_likelihood`, …). https://python.arviz.org/en/stable/schema/schema.html
- **Martin, Kumar, Lao, *Bayesian Modeling and Computation in Python*, Ch 2 — "Exploratory Analysis of Bayesian Models"** — the course-aligned ArviZ walkthrough of `az.summary`, trace/rank, divergences, PPC. Free online. https://bayesiancomputationbook.com/notebooks/chp_02.html
- **PyMC `pymc.sample` / `pymc.NUTS` / `pymc.init_nuts` API docs** — verify every knob and default (`target_accept`, `max_treedepth`, the `init` strings). https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.sample.html
- **(Intuition) Chi Feng's interactive MCMC demo** — watch a sampler diverge in a funnel in real time. https://chi-feng.github.io/mcmc-demo/

---

## ➡️ What's next

You can now look a fitted model in the eye and decide whether to trust it — count divergences, read $\hat R$ and ESS, localize a funnel, and reparameterize it away. But diagnostics are only one **stage** of a larger process. **Chapter 08 — The Bayesian Workflow** zooms out and shows you where these checks live in the full loop: prior predictive checks *before* you fit, the diagnostic gate you just learned, posterior predictive checks and **simulation-based calibration (SBC)** to validate the inference itself, and the iterative model-expansion cycle that Betancourt and Gelman et al. formalized. The diagnostics you mastered here become the quality gate that every step of that workflow has to pass — and the centered/non-centered move you just learned becomes a reflex you'll reach for in **Chapter 10 (Hierarchical & Multilevel Models)**, where funnels are not a teaching example but a daily fact of life.
