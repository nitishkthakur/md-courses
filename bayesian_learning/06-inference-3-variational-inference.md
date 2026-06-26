# Chapter 06 — Inference III: Variational Inference

### Turning posterior inference into optimization — and learning exactly where that bargain costs you.

> *A note from me to you.*
>
> *In Chapter 05 we built the gold standard. NUTS explores the posterior by simulating a frictionless puck rolling over the log-probability surface, and if you tune it and read its diagnostics, it converges to the truth. It is honest. But honesty is expensive: every gradient evaluation touches your whole dataset, and for a model with a million rows or ten thousand parameters, "let it run overnight" stops being a joke and becomes your actual workflow.*
>
> *Variational inference (VI) makes a different bargain. Instead of drawing samples from the true posterior, it asks: "Of all the simple distributions I can write down in closed form — say, Gaussians — which one is closest to the posterior?" Then it finds that closest one by gradient descent, the same machinery that trains your neural nets. Inference becomes optimization. It is fast, it scales to data that NUTS chokes on, and it gives you a usable answer in seconds.*
>
> *Here is the catch, and it is the whole reason this chapter exists: **VI gives you the right shape of the wrong distribution.** The standard flavor systematically reports posteriors that are too narrow. Your 95% intervals will be too tight, your significance too confident, your scale parameters too small. If you don't know this is happening, VI will quietly lie to you and you'll ship overconfident conclusions. The painful mistake this chapter prevents is trusting a fast answer that is precisely, measurably, wrong in a known direction.*
>
> *So we are going to do two things. First, understand VI deeply enough that you know exactly why it underestimates variance — it falls straight out of one sign in one equation. Second, and more important, learn to **catch it in the act**: overlay VI against NUTS, compute the Pareto-$\hat{k}$ diagnostic on the importance weights, and decide for yourself whether the speed was worth it. By the end you'll reach for VI deliberately — for massive data, for fast iteration, for initializing MCMC — and never by accident.*

---

## What you'll be able to do after this chapter

- Explain the **big idea** of VI: cast posterior inference as an optimization problem over a family of approximating distributions $q$.
- Derive the **ELBO** from the log-evidence and show that maximizing it is identical to minimizing $\mathrm{KL}(q\|p)$.
- Say precisely **why the reverse-KL direction** ($\mathrm{KL}(q\|p)$, not $\mathrm{KL}(p\|q)$) makes VI **mode-seeking** and **variance-underestimating** — and contrast with the mass-covering forward KL.
- Understand the **mean-field assumption** and the price you pay for it (ignored posterior correlations).
- Run **ADVI** (mean-field) and **full-rank ADVI** in PyMC with `pm.fit(method=...)`, read the ELBO trajectory, and draw samples with `approx.sample`.
- Place **Pathfinder** and **SVGD** in the landscape, and know when each beats plain ADVI.
- **Detect** VI failure: overlay against NUTS, compute PSIS $\hat{k}$ on $\log p - \log q$, and read the verdict.
- Decide **when VI is the right tool** — and when you must fall back to NUTS for the final answer.

---

## 1. The big idea: inference as optimization

Let me set the stage with the object we have been chasing since Chapter 01. Bayes' theorem says

$$
p(\theta \mid y) \;=\; \frac{p(y \mid \theta)\, p(\theta)}{p(y)}, \qquad p(y) = \int p(y\mid\theta)\,p(\theta)\,d\theta.
$$

The numerator is easy — it's just the likelihood times the prior, both of which you wrote down yourself. The villain, as always, is the denominator $p(y)$, the **marginal likelihood** or **evidence**: an integral over the entire parameter space that is intractable for essentially every model you'll ever care about. Every inference method in this course is, at heart, a strategy for getting at the posterior *without* computing that integral.

- **Grid approximation (Ch 04)** evaluates the unnormalized posterior on a lattice and normalizes by brute-force summation. Exact-ish in 1–2 dimensions, dead by 4 (the curse of dimensionality: cost grows like $K^d$).
- **Laplace / quadratic approximation (Ch 04)** fits a single Gaussian at the posterior mode using the curvature (Hessian) there. Cheap, but it's a one-shot local fit — no exploration.
- **MCMC / NUTS (Ch 05)** draws correlated samples whose stationary distribution *is* the posterior. Asymptotically exact. The gold standard, and expensive.
- **Variational inference (this chapter)** does something philosophically distinct from all three.

> 💡 **Intuition:** Sampling methods *represent* the posterior with a cloud of points. Variational inference *replaces* the posterior with a tractable stand-in distribution $q(\theta)$, chosen from a family you pick in advance (e.g. "all Gaussians"), and tuned so that $q$ is as close as possible to the true $p(\theta\mid y)$. You then use $q$ — which has a closed form, samples instantly, and gives you means and quantiles cheaply — as a proxy for the posterior.

Here is the reframing that makes the whole field click. Define a family of candidate distributions $q(\theta; \phi)$ parameterized by **variational parameters** $\phi$. For a mean-field Gaussian, $\phi$ is just a vector of means and a vector of standard deviations. The job of VI is:

$$
q^\star = \arg\min_{q \in \mathcal{Q}} \; D\big(q(\theta) \,\|\, p(\theta\mid y)\big),
$$

where $D$ is some measure of discrepancy between distributions. **That is an optimization problem.** No Markov chains, no acceptance ratios, no autocorrelation. Just: pick a divergence, pick a family, and run gradient descent on $\phi$. The same Adam optimizer that trains your transformer can fit your Bayesian posterior.

> 📜 **Citation/Origin:** The modern, automatic form of this idea is **Automatic Differentiation Variational Inference (ADVI)**, Kucukelbir, Tran, Ranganath, Gelman & Blei, *JMLR* 18(14):1–45, 2017 ([jmlr.org/papers/v18/16-107.html](https://jmlr.org/papers/v18/16-107.html)). The best single conceptual reference is Blei, Kucukelbir & McAuliffe, *"Variational Inference: A Review for Statisticians"*, JASA 2017 ([arXiv:1601.00670](https://arxiv.org/abs/1601.00670)). VI itself goes back to the 1990s mean-field methods from statistical physics and machine learning.

This trade is seductive. It is also where careers go sideways, because the choice of $D$ and the choice of $\mathcal{Q}$ are not free — they bake in biases that you must understand cold. Let's earn that understanding.

---

## 2. KL divergence and which direction we choose

The discrepancy $D$ that VI uses is the **Kullback–Leibler divergence**. For two densities $q$ and $p$:

$$
\mathrm{KL}(q \,\|\, p) \;=\; \mathbb{E}_{q}\!\left[\log \frac{q(\theta)}{p(\theta)}\right] \;=\; \int q(\theta)\,\log\frac{q(\theta)}{p(\theta)}\,d\theta.
$$

A few facts to keep in your pocket:

- $\mathrm{KL}(q\|p) \ge 0$ always, with equality **iff** $q = p$ almost everywhere (Gibbs' inequality). So it's a sensible "distance from $q$ to $p$" — zero when they match, positive otherwise.
- It is **not symmetric**: $\mathrm{KL}(q\|p) \ne \mathrm{KL}(p\|q)$ in general. This asymmetry is not a technicality. It is the single most consequential design choice in variational inference, and it is the reason VI lies to you in a *predictable* direction.

VI minimizes the **reverse KL**, $\mathrm{KL}(q\|p)$ — the expectation is taken under $q$, the approximation, with the true posterior $p$ in the denominator. (The "forward" KL would be $\mathrm{KL}(p\|q)$, expectation under the true posterior.) We use reverse KL not because it is more correct but because it is **computable**: the expectation $\mathbb{E}_q[\cdot]$ is over our own tractable $q$, which we can sample from and differentiate through. The forward KL would require expectations under the very posterior $p$ we cannot touch.

> 🧮 **The math — why reverse KL is mode-seeking and variance-underestimating.**
> Look at the integrand of $\mathrm{KL}(q\|p) = \int q \,\log(q/p)$. Consider two regions:
>
> 1. **Where $p \approx 0$ but $q > 0$** (the approximation puts mass where the posterior has none): then $\log(q/p) \to +\infty$, and because $q > 0$ there, this region contributes a *large positive penalty*. Reverse KL **hates** putting mass outside the support of $p$.
> 2. **Where $q \approx 0$ but $p > 0$** (the posterior has mass the approximation ignores): then $q \log(q/p) \to 0$ (since $x\log x \to 0$ as $x \to 0$). Reverse KL is **indifferent** to ignoring mass that $p$ has.
>
> The optimizer therefore makes $q$ *retreat*: it squeezes $q$ into a region where $p$ is reliably large and refuses to spill over the edges. With a multimodal $p$, $q$ will typically grab **one mode** and ignore the rest — it is cheaper to nail one peak than to stretch across a low-density valley and pay the penalty in region 1. This is **mode-seeking** behavior, and its concrete consequence is **variance underestimation**: $q$ ends up narrower than $p$.

The mirror image is worth one paragraph. **Forward KL**, $\mathrm{KL}(p\|q) = \int p\log(p/q)$, swaps the roles. Now $\log(p/q)\to+\infty$ wherever $p>0$ but $q\approx0$ — so forward KL hates *ignoring* posterior mass and forces $q$ to **cover** all of $p$, even if that means spreading across the valleys between modes. Forward KL is **mass-covering** and tends to *overestimate* variance. It's the flavor behind Expectation Propagation and moment-matching methods. We don't use it in standard VI because the expectation is under the intractable $p$, but knowing it exists sharpens your picture: VI's narrowness is a *choice of divergence direction*, not a law of nature.

> ⚠️ **Pitfall:** "VI is approximate, so its errors are random noise I can average away." No. The error has a **direction and a sign**: standard (reverse-KL) VI is biased *toward narrowness*. Running it longer, or with more samples, does not fix this — it converges faithfully to the best-but-still-too-narrow $q$ in your family. The only cures are a richer family ($\mathcal{Q}$) or a different method entirely.

Here is the picture every Bayesian carries in their head. A correlated, slightly skewed 2-D posterior, with the mean-field Gaussian VI approximation drawn over it:

```
   True posterior p(θ)              Mean-field VI q(θ)
   (correlated, broad)              (axis-aligned, narrow)

        ╭────────╮                       ┌──┐
      ╭─╯ ▓▓▓▓▓▓ ╰─╮                     │▓▓│
    ╭─╯ ▓▓▓▓▓▓▓▓▓▓ ╰─╮          →        │▓▓│   ← narrower in every
   │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓  │                  └──┘     direction; axes
    ╰─╮ ▓▓▓▓▓▓▓▓▓▓ ╭─╯                            aligned, so the tilt
      ╰─╮ ▓▓▓▓▓▓ ╭─╯                              (correlation) is lost
        ╰────────╯
```

It sits inside the true posterior, axis-aligned (no tilt = no correlation), and tight. Burn that image in. It is what "VI lies" looks like.

---

## 3. The ELBO, derived from scratch

We said VI minimizes $\mathrm{KL}(q\|p)$. But there's an immediate problem: $\mathrm{KL}(q\|p)$ contains the true posterior $p(\theta\mid y) = p(\theta,y)/p(y)$, which contains the intractable evidence $p(y)$. If we needed $p(y)$ to optimize, VI would be no easier than the integral we're fleeing. The resolution is the **Evidence Lower BOund (ELBO)**, and it's a two-line derivation you should be able to reproduce on a whiteboard.

> 🧮 **The math — deriving the ELBO.**
> Start from the log-evidence and multiply by $1 = \int q(\theta)\,d\theta$ inside, then use $p(\theta\mid y) = p(\theta,y)/p(y)$:
>
> $$
> \log p(y) = \log p(y) \int q(\theta)\,d\theta = \int q(\theta)\,\log p(y)\,d\theta = \mathbb{E}_q\big[\log p(y)\big].
> $$
>
> Now write $\log p(y) = \log\dfrac{p(\theta,y)}{p(\theta\mid y)}$ and split the log into a ratio with $q$ inserted top and bottom:
>
> $$
> \log p(y) = \mathbb{E}_q\!\left[\log \frac{p(\theta,y)}{p(\theta\mid y)}\right]
> = \mathbb{E}_q\!\left[\log \frac{p(\theta,y)}{q(\theta)}\right] + \mathbb{E}_q\!\left[\log \frac{q(\theta)}{p(\theta\mid y)}\right].
> $$
>
> The second term is exactly $\mathrm{KL}(q\|p(\theta\mid y))$. The first term we name the **ELBO**:
>
> $$
> \boxed{\;\log p(y) \;=\; \underbrace{\mathbb{E}_q\!\left[\log \frac{p(\theta,y)}{q(\theta)}\right]}_{\textstyle \mathrm{ELBO}(q)} \;+\; \underbrace{\mathrm{KL}\big(q(\theta)\,\|\,p(\theta\mid y)\big)}_{\textstyle \ge\, 0}\;}
> $$

Sit with that identity, because it does all the work. The left side, $\log p(y)$, is a **fixed constant** — it does not depend on $q$ at all. The right side is a sum of two terms that *do* depend on $q$. Since they must always add up to the same constant, **pushing one up pushes the other down by exactly the same amount.** And since $\mathrm{KL} \ge 0$:

$$
\mathrm{ELBO}(q) \;=\; \log p(y) - \mathrm{KL}(q\|p) \;\le\; \log p(y).
$$

So the ELBO is a **lower bound on the log-evidence** (that's the name), and the gap between them is precisely the KL we want to minimize. The chain of logic, in plain English:

> **Maximizing the ELBO** $\;\Longleftrightarrow\;$ **minimizing** $\mathrm{KL}(q\|p)$ $\;\Longleftrightarrow\;$ **making $q$ as close as possible to the true posterior** — all without ever computing $p(y)$, because $p(y)$ is the constant we're sliding the bound up toward.

In ASCII, the decomposition you'll recite forever:

```
log p(y)  =  ELBO(q)  +  KL(q || p)        [constant = bound + gap]
            ^^^^^^^^     ^^^^^^^^^^^
            maximize     minimize           (they move in lockstep)
```

The ELBO has two equivalent rewrites, each illuminating a different intuition. Expand $p(\theta,y) = p(y\mid\theta)p(\theta)$:

$$
\mathrm{ELBO}(q) = \underbrace{\mathbb{E}_q[\log p(y\mid\theta)]}_{\text{expected log-likelihood}} \;-\; \underbrace{\mathrm{KL}(q(\theta)\,\|\,p(\theta))}_{\text{prior regularizer}}.
$$

> 💡 **Intuition:** The ELBO is a **fit-versus-complexity tug-of-war**, and it should feel familiar from regularized ML. The first term rewards $q$ for concentrating on parameters that explain the data well (high expected log-likelihood). The second term penalizes $q$ for drifting too far from the prior. Maximize their difference and you get the classic Bayesian compromise — fit the data, but don't stray further from your prior beliefs than the data justify. It is the variational echo of the posterior itself.

The other rewrite, using entropy $\mathbb{H}[q] = -\mathbb{E}_q[\log q]$:

$$
\mathrm{ELBO}(q) = \mathbb{E}_q[\log p(\theta, y)] + \mathbb{H}[q].
$$

Here the second term, the entropy of $q$, *rewards spread*. This is the only thing fighting the variance collapse — without the entropy term, $q$ would gleefully shrink to a spike at the mode. So why does VI still come out too narrow? Because the entropy reward is weaker than the reverse-KL penalty for spilling mass outside $p$. The forces don't balance at the true posterior width; they balance somewhere tighter.

> 🔧 **In practice:** PyMC does not maximize the ELBO directly; it **minimizes the negative ELBO** (a "loss," exactly like a neural-net training loss). The trajectory of that loss across iterations is stored in `approx.hist`, and watching it flatten is your first, crudest convergence check. We'll plot it in the worked example.

---

## 4. The mean-field assumption and what it costs

We have a clean objective (maximize the ELBO) but we still have to pick the **family** $\mathcal{Q}$ that $q$ lives in. The most common, and PyMC's default, is the **mean-field** family: $q$ factorizes into independent one-dimensional pieces, one per (unconstrained) parameter:

$$
q(\theta) = \prod_{i=1}^{d} q_i(\theta_i) = \prod_{i=1}^{d} \mathrm{Normal}(\theta_i; \mu_i, \sigma_i^2).
$$

Each parameter gets its own Gaussian with its own mean $\mu_i$ and standard deviation $\sigma_i$. The full set of variational parameters is $\phi = \{\mu_1,\dots,\mu_d,\sigma_1,\dots,\sigma_d\}$ — just $2d$ numbers, which is why mean-field is fast and memory-light even in high dimensions.

The word "independent" is where the cost hides. A mean-field $q$ has a **diagonal covariance matrix**: it asserts that, under the approximation, the parameters are mutually uncorrelated. But real posteriors are riddled with correlations. In a regression, the intercept and slope trade off (raise the slope, lower the intercept to keep the line through the data). In a hierarchical model, the group-level mean and the group-level scale are entangled. Mean-field throws all of that away.

> 🧮 **The math — the double cost of mean-field.** When the true posterior is a correlated Gaussian and you fit the best *diagonal* Gaussian by reverse KL, two things happen simultaneously:
> 1. **Correlations vanish.** Your $q$ is axis-aligned; the posterior's tilt is invisible. Joint statements ("$\theta_1$ and $\theta_2$ are anti-correlated") are simply unavailable.
> 2. **Marginal variances shrink.** This is the subtle, dangerous one. The reverse-KL-optimal diagonal Gaussian fits the **conditional** width of the posterior, not the **marginal** width. For a 2-D Gaussian with correlation $\rho$, the mean-field marginal standard deviations come out a factor of $\sqrt{1-\rho^2}$ too small. At $\rho = 0.9$ that's $\sqrt{1-0.81}\approx 0.44$ — the reported standard deviation is **less than half** the truth. Your 95% intervals shrink by the same factor.

That second point is the one that bites in production. It is not enough to say "mean-field loses correlations"; the loss of correlation *feeds back into* an underestimate of every marginal variance. The stronger the true correlations, the more overconfident your VI marginals. This is why a logistic regression with nearly-orthogonal standardized predictors (weak posterior correlation) is a *good* case for mean-field VI, while a hierarchical funnel (extreme correlation between a scale and the things it scales) is a *catastrophe* for it.

> ⚠️ **Pitfall:** Standardizing your predictors (Ch 03's default) is not just good prior hygiene — it also *decorrelates the posterior*, which directly shrinks the $\sqrt{1-\rho^2}$ penalty and makes mean-field VI far more trustworthy. If you're going to use VI, standardize first. It's the cheapest accuracy you'll ever buy.

The remedy that stays within Gaussians is **full-rank** VI: let $q = \mathrm{Normal}(\mu, \Sigma)$ with a *dense* covariance $\Sigma = LL^\top$ (parameterized by a Cholesky factor $L$ to keep it positive-definite). Full-rank can represent any correlation structure a Gaussian can, so it recovers the tilt and the correct marginal widths — *if the posterior is genuinely Gaussian*. The price is parameter count: $\Sigma$ has $O(d^2)$ free entries versus mean-field's $O(d)$, so full-rank is slower and can be unstable in high dimensions. And note the standing caveat: even full-rank VI minimizes reverse KL, so on a **non-Gaussian** posterior (skewed, heavy-tailed, the funnel's neck) it will still underestimate variance — just less than mean-field does.

| Family | $q$ | # variational params | Captures correlations? | Cost | Still underestimates variance? |
|---|---|---|---|---|---|
| **Mean-field (ADVI)** | $\prod_i \mathrm{N}(\mu_i,\sigma_i^2)$ | $2d$ | No (diagonal) | Cheap | Yes — most |
| **Full-rank ADVI** | $\mathrm{N}(\mu, LL^\top)$ | $d + d(d{+}1)/2$ | Yes (dense $\Sigma$) | $O(d^2)$, slower | Yes — less, only if non-Gaussian |
| **Normalizing-flow VI** | flow-transformed base | many | Yes + non-Gaussian shape | Expensive | Much less |
| **NUTS (Ch 05)** | exact samples | — | Yes, exactly | Most | No (asymptotically exact) |

---

## 5. ADVI: making it automatic

We have the objective (ELBO) and the family (Gaussian, mean-field or full-rank). What turns this into a one-line `pm.fit(...)` call you can run on *any* model without hand-deriving anything is **ADVI**. There are three moving parts, and understanding them tells you exactly when ADVI will misbehave.

**Part 1 — Transform constrained parameters to the whole real line.** A Gaussian $q$ lives on $\mathbb{R}^d$: it puts mass everywhere from $-\infty$ to $+\infty$. But your model has constrained parameters — a standard deviation $\sigma > 0$, a probability $p \in (0,1)$, a simplex. You cannot lay an unbounded Gaussian over a positive-only parameter without leaking mass into negative, illegal territory. So ADVI first **bijects** every constrained parameter to an unconstrained one: $\sigma \mapsto \log\sigma \in \mathbb{R}$, $p \mapsto \mathrm{logit}(p) \in \mathbb{R}$, etc. PyMC already maintains these transforms internally (it's the same machinery NUTS uses), so this is free. The Gaussian $q$ is fit in the **unconstrained space**, then samples are pushed back through the inverse transform to the natural scale.

> 💡 **Intuition:** This is why VI (and Laplace, Ch 04) can produce a *reasonable* posterior for a positive scale parameter even though "a Gaussian on $\sigma$" sounds absurd. The Gaussian lives on $\log\sigma$, where the posterior genuinely is more bell-shaped; exponentiating turns it into a right-skewed, strictly-positive distribution on $\sigma$. The skew is recovered for free by the transform.

**Part 2 — Estimate the ELBO and its gradient with the reparameterization trick.** The ELBO is an expectation $\mathbb{E}_q[\dots]$ we cannot compute in closed form for a general model. We estimate it by Monte Carlo: draw a few samples from $q$ and average. But we need to differentiate the result with respect to $\phi = (\mu, \sigma)$ — and the samples themselves depend on $\phi$, so naive sampling blocks the gradient. The fix is the **reparameterization trick**: instead of drawing $\theta \sim \mathrm{Normal}(\mu, \sigma^2)$ directly, draw a *parameter-free* noise variable $\epsilon \sim \mathrm{Normal}(0,1)$ and set

$$
\theta = \mu + \sigma \cdot \epsilon.
$$

Now the randomness ($\epsilon$) is divorced from the parameters ($\mu,\sigma$), so the gradient $\nabla_\phi$ flows straight through $\theta$ into the ELBO. This is the exact same trick that powers variational autoencoders, and it's what makes the gradients **low-variance** enough for stochastic optimization to actually converge.

> 📜 **Citation/Origin:** Casting these as automatic, differentiable Monte-Carlo ELBO gradients over a transformed space is the contribution of Kucukelbir et al.'s ADVI paper (2017). Before ADVI, every model needed bespoke, hand-derived coordinate-ascent updates (CAVI). ADVI made VI as push-button as autodiff made backprop.

**Part 3 — Stochastic gradient ascent.** With unbiased, low-variance gradient estimates in hand, we just run a stochastic optimizer (Adam, Adagrad-window, …) on the variational parameters $\phi$, maximizing the ELBO (equivalently, minimizing the negative ELBO loss). Because each step uses only a Monte-Carlo estimate of the gradient — and, with **minibatch ADVI**, only a random subset of the *data* — VI scales to datasets where a single full-data NUTS gradient is unaffordable. That's the scaling payoff.

> 🔧 **In practice:** The three parts map to three failure modes. Bad **transforms** are handled for you. Noisy **gradients** mean you should raise the number of optimization steps `n` or lower the learning rate. A multimodal **target** means stochastic gradient ascent can get stuck near one mode — exactly the mode-seeking pathology, now operationalized. None of these are bugs; they're the cost of the bargain.

### The PyMC API for ADVI

Here is the canonical pattern. Everything returns objects that flow straight into ArviZ.

```python
import pymc as pm
import arviz as az
import numpy as np
import matplotlib.pyplot as plt

RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)

with model:                                   # any PyMC model
    # Mean-field ADVI (the default). n = number of optimization iterations.
    approx = pm.fit(n=30_000, method="advi", random_seed=RANDOM_SEED)

# Draw samples from the fitted q -> a standard InferenceData object
idata_advi = approx.sample(2000)              # use with az.summary, az.plot_*, etc.

# The negative-ELBO trajectory (the "loss"): eyeball convergence
plt.plot(approx.hist)
plt.xlabel("iteration"); plt.ylabel("negative ELBO (loss)")

# The fitted variational parameters (mean-field: per-parameter mean & sd, unconstrained scale)
approx.mean.eval()        # vector of means  (μ) on the unconstrained space
approx.std.eval()         # vector of std devs (σ) on the unconstrained space
```

The method strings PyMC accepts are `"advi"` (mean-field, default), `"fullrank_advi"`, `"svgd"`, and `"asvgd"`. There's also an object-oriented form that exposes `.refine`, which continues optimizing the *same* approximation — handy when the loss hasn't flattened and you don't want to start over:

```python
with model:
    advi = pm.ADVI()                          # or pm.FullRankADVI()
    approx = advi.fit(30_000, random_seed=RANDOM_SEED)
    advi.refine(20_000)                        # keep optimizing the SAME q for 20k more steps
```

A convergence callback stops you early once the variational parameters stop moving — cheaper than guessing `n`:

```python
from pymc.variational.callbacks import CheckParametersConvergence

with model:
    approx = pm.fit(
        n=50_000, method="advi",
        callbacks=[CheckParametersConvergence(diff="absolute")],  # also "relative"
        random_seed=RANDOM_SEED,
    )
```

> 🩺 **Diagnostic — reading `approx.hist`.** Plot the negative ELBO against iteration. A **healthy** run drops fast at first, then flattens into a noisy plateau (the noise is the Monte-Carlo gradient estimate; it never goes to zero). If the curve is **still descending** at the end, you stopped too early — `refine` or raise `n`. If it's **wildly oscillating** or shoots to $\pm\infty$ early, your data is likely unstandardized or your priors are extreme; standardize and re-try. A flat plateau means the *optimization* converged — it does **not** mean the *approximation is good*. Those are two different questions, and §8 is about the second one.

### Full-rank ADVI in PyMC

Switching to full-rank is a one-word change. Use it when you suspect strong posterior correlations and the dimension is modest enough to afford the dense covariance:

```python
with model:
    approx_fr = pm.fit(n=30_000, method="fullrank_advi", random_seed=RANDOM_SEED)
idata_fr = approx_fr.sample(2000)
```

Full-rank will recover the *tilt* (correlations) and therefore the correct marginal widths **when the posterior is close to Gaussian**. It cannot fix non-Gaussianity, and in high dimensions the $O(d^2)$ covariance makes it slow and sometimes numerically fragile. A common, pragmatic ladder: try mean-field; if it disagrees badly with a short NUTS run, try full-rank; if that still disagrees, the posterior isn't Gaussian and you should be running NUTS for real.

---

## 6. Beyond Gaussian q: normalizing flows and SVGD

ADVI's ceiling is that $q$ is a Gaussian (diagonal or dense). If the true posterior is skewed, heavy-tailed, banana-shaped, or multimodal, *no* Gaussian can match it, and you're stuck with the best-fitting wrong shape. Two families of method break the Gaussian ceiling.

**Normalizing-flow VI (NFVI).** The idea is elegant: start with a simple base distribution (a standard Gaussian) and push it through a chain of **invertible, differentiable maps** $f_K \circ \cdots \circ f_1$. Each map bends and stretches the density; the change-of-variables formula tracks how, via the log-determinant of each map's Jacobian:

$$
\log q(\theta) = \log q_0(z_0) - \sum_{k=1}^{K} \log\left|\det \frac{\partial f_k}{\partial z_{k-1}}\right|, \qquad \theta = f_K\circ\cdots\circ f_1(z_0).
$$

Because the maps are learnable, a flow can sculpt the base Gaussian into a banana, a skew, even multiple modes — a *strictly more expressive* family than ADVI, fit by the same ELBO/SGD machinery. The cost is a much larger parameter count and trickier optimization.

> ⚠️ **Pitfall:** PyMC historically shipped an `NFVI` / `flows` API, but it is **legacy and effectively unmaintained** — do not reach for a specific current PyMC flow call expecting it to work. If you want a concrete, supported normalizing-flow guide today, use **NumPyro's autoguides**: `AutoIAFNormal` (inverse autoregressive flow) or `AutoBNAFNormal` (block neural autoregressive flow). `# verify in current NumPyro docs` before relying on exact symbols.

```python
# pseudocode — NumPyro normalizing-flow VI (the supported route for flows today)
# verify in current NumPyro docs: num.pyro.ai/en/stable/autoguide.html
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoIAFNormal   # a flow-based guide
guide = AutoIAFNormal(model)                          # vs AutoNormal (mean-field), AutoMultivariateNormal (full-rank)
svi = SVI(model, guide, numpyro.optim.Adam(1e-3), Trace_ELBO())
svi_result = svi.run(rng_key, num_steps=5000)
```

These NumPyro autoguides are the JAX bridge we'll lean on in **Chapter 17 (Scaling and Advanced Computation)**: `AutoNormal` ≡ mean-field ADVI, `AutoMultivariateNormal` ≡ full-rank, `AutoLaplaceApproximation` ≡ the Laplace approximation from Chapter 04, and the flow guides for when Gaussians fail.

**SVGD — Stein Variational Gradient Descent.** A completely different idea: represent $q$ not by parameters but by a *set of particles* $\{\theta^{(1)},\dots,\theta^{(m)}\}$. The particles start scattered and evolve under a deterministic, kernelized gradient flow that provably decreases $\mathrm{KL}(q\|p)$ at each step. Two forces act on each particle: the log-posterior gradient pulls it toward high-density regions, while a **repulsive kernel term** pushes particles apart so they don't all collapse onto the single mode. Because $q$ is just the particle cloud — no Gaussian assumption — SVGD can represent non-Gaussian and even multimodal posteriors.

```python
with model:
    approx_svgd = pm.fit(
        n=300, method="svgd",
        inf_kwargs=dict(n_particles=1000),
        obj_optimizer=pm.sgd(learning_rate=0.01),
        random_seed=RANDOM_SEED,
    )
```

> 🔧 **In practice:** Treat SVGD (and its amortized cousin `method="asvgd"`) as "beyond-Gaussian VI worth knowing about," not a default. They're far less battle-tested than ADVI or NUTS, sensitive to the kernel and particle count, and can be slow. The repulsion term is what keeps SVGD from collapsing the way mean-field ADVI does — a nice conceptual contrast — but for production work, if a Gaussian VI isn't enough, the modern first move is **Pathfinder** or just NUTS. (`# verify "asvgd" is still wired in your installed PyMC before relying on it.`)

---

## 7. Pathfinder: the best of both worlds for initialization

If you remember one *modern* method from this chapter, make it **Pathfinder**. It sits in the sweet spot between Laplace (Ch 04) and full VI, it's much faster than NUTS, more robust than plain ADVI, and it makes an excellent **NUTS initializer**.

> 📜 **Citation/Origin:** Zhang, Carpenter, Gelman & Vehtari, *"Pathfinder: Parallel quasi-Newton variational inference,"* JMLR 23(306):1–49, 2022 ([jmlr.org/papers/v23/21-0889.html](https://jmlr.org/papers/v23/21-0889.html), [arXiv:2108.03782](https://arxiv.org/abs/2108.03782)). It also ships in Stan ([mc-stan.org/docs/reference-manual/pathfinder.html](https://mc-stan.org/docs/reference-manual/pathfinder.html)).

Here's the mechanism, and it's genuinely clever. Run an **L-BFGS** optimizer from a random starting point, climbing toward the MAP. At every iterate along that optimization *path*, L-BFGS already maintains a cheap low-rank estimate of the inverse Hessian — so you can read off a **Laplace-like Gaussian approximation at every point on the path**, essentially for free. Now the key insight: the Gaussian *right at the mode* (what plain Laplace gives you) is often too narrow, but somewhere **earlier on the path**, before the optimizer has fully tightened into the peak, there's an iterate whose Gaussian best matches the posterior's spread. Pathfinder estimates the KL divergence to the posterior at each iterate and **picks the single best one**. Then it draws from that chosen Gaussian and applies **Pareto-smoothed importance resampling (PSIS/PSIR)** to correct the draws toward the true posterior.

> 💡 **Intuition:** Plain Laplace asks "what Gaussian fits the very tip of the mountain?" and often gets a too-pointy answer. Pathfinder watches the whole climb and asks "at which moment along the way did my Gaussian best describe the mountain's actual girth?" — then corrects the leftover error with importance weights. It's Laplace that shopped around the optimization trajectory for a better fit.

**Multi-path Pathfinder** runs several L-BFGS climbs from different random starts in parallel and PSIR-combines their draws. This is its defense against multimodality and bad initializations: if one path stumbles into a poor mode, the importance resampling down-weights it in favor of the paths that found good ones. Raise `num_paths` and `jitter` for tougher posteriors.

### The PyMC API for Pathfinder

Pathfinder lives in `pymc_extras` (the package formerly called `pymc-experimental`; `import pymc_extras as pmx`). Two equivalent entry points:

```python
import pymc_extras as pmx
from pymc_extras.inference import fit_pathfinder   # verify in current pymc_extras docs

with model:
    # High-level wrapper (dispatches to fit_pathfinder):
    idata_path = pmx.fit(method="pathfinder", num_draws=1000, jitter=12,
                         random_seed=RANDOM_SEED)

    # OR the direct function, with the full knob set:
    idata_path = fit_pathfinder(
        num_paths=4,                       # parallel L-BFGS runs; raise for multimodal posteriors
        num_draws=1000,                    # total posterior draws returned
        importance_sampling="psis",        # {"psis"(default), "psir", "identity", None}
        inference_backend="pymc",          # or "blackjax"
        jitter=12.0,                       # init spread; raise it (and num_paths) for tough posteriors
        jacobian_correction=True,          # KEEP True — disabling it inflates Pareto-k spuriously
        random_seed=RANDOM_SEED,
    )
```

The return value is **an InferenceData object, just like `pm.sample()`** — so `az.summary(idata_path)` and `az.plot_forest([idata_nuts, idata_path], ...)` work immediately. That interoperability is the whole point: Pathfinder slots into your existing ArviZ workflow with zero ceremony.

> 🔧 **In practice — Pathfinder as a NUTS initializer.** Pathfinder's biggest day-to-day win isn't standalone inference; it's getting NUTS off to a flying start. A Pathfinder approximation gives NUTS a good initial point *and* a sensible mass matrix, which can dramatically shorten warmup on hard models. PyMC exposes this directly:
> ```python
> with model:
>     idata = pm.sample(nuts_sampler="nutpie", init="pathfinder", random_seed=RANDOM_SEED)
>     # init="pathfinder" warms NUTS up from a Pathfinder fit. # verify in current PyMC docs
> ```
> Even when you don't trust VI for the final answer, you can trust it to *point NUTS in the right direction*. This is the single most reliable use of VI in a serious workflow.

> ⚠️ **Pitfall:** Pathfinder can hand you a *confident wrong answer* if every path lands in the same bad mode or the Gaussian family just can't describe your posterior (the funnel's neck, again). Two tells: (1) high **Pareto-$\hat{k}$** everywhere in its importance-sampling diagnostic, and (2) disagreement with a short NUTS run. If you see high $\hat{k}$, first confirm `jacobian_correction=True` (setting it `False` produces spuriously high $\hat{k}$); if it's still high, the Gaussian family is inadequate and you must use NUTS.

Where does Pathfinder sit relative to everything else? It's the modern bridge across the whole arc of Chapters 04–06:

```
grid (Ch04)  →  Laplace/QUAP (Ch04)  →  ADVI / full-rank (Ch06)  →  Pathfinder (Ch06)  →  NUTS (Ch05)
exact-ish      one Gaussian at         best Gaussian by SGD          best Gaussian along       asymptotically
1–2 dims       the mode, cheap         ELBO, scales to big data      an L-BFGS path + PSIS     exact, gold standard
                                                                      (great NUTS init)
```

---

## 8. How VI lies — and how to catch it

This is the most important section in the chapter. Everything above was setup; this is the judgment that separates someone who *uses* VI from someone who gets *burned* by it. VI fails in three characteristic ways, and there are two reliable ways to detect failure.

### The three lies

1. **Underdispersed posteriors (too-narrow intervals).** The headline failure, straight from reverse KL. Every marginal standard deviation comes out too small; every credible interval is too tight; every "this effect is clearly nonzero" is more confident than the data warrant. A VI 95% interval might be a true 70% interval. In decision-making this is the dangerous one — you'll act as if you know more than you do.
2. **Missed modes.** Mode-seeking means a multimodal posterior gets collapsed onto one peak. VI will report a crisp unimodal answer and give you *no hint* that a second, equally-good explanation of the data exists. The ELBO trajectory will look perfectly converged. Silence is the failure mode.
3. **Lost correlations and missed geometry.** Mean-field flattens the joint structure, so any conclusion that depends on the *joint* posterior — a contrast, a sum of coefficients, a hierarchical funnel's neck — can be badly wrong even when each marginal looks individually plausible.

> ⚠️ **Pitfall:** None of these announce themselves. A converged ELBO, a clean `az.summary`, tidy-looking marginals — VI can show you all of that *while being wrong*. Optimization convergence (the loss flattened) and approximation quality (q is close to p) are **different questions**. You must check the second one explicitly. It will never check itself.

### Detection method 1: overlay against NUTS

The most direct check, and the one you should run every single time you consider trusting VI for inference: **run a (possibly short) NUTS chain and plot the two posteriors on top of each other.** ArviZ makes this a one-liner because VI and NUTS both produce InferenceData.

```python
# Overlay marginals: where VI is narrower than NUTS, you can SEE the lie.
az.plot_forest([idata_nuts, idata_advi], model_names=["NUTS", "ADVI"],
               combined=True, figsize=(8, 5))

# Or per-parameter density overlays:
az.plot_density([idata_nuts, idata_advi], data_labels=["NUTS", "ADVI"],
                shade=0.2)
```

> 🩺 **Diagnostic — reading the overlay.** On the **forest plot**, each parameter gets a row with two intervals stacked. The NUTS interval is your truth; the ADVI interval almost always sits **inside** it, shorter and centered at roughly the same place. The amount of shrinkage *is* the variance underestimation, and now you can **measure** it (e.g. ratio of interval widths). On the **density overlay**, you'll see the ADVI curve as a taller, skinnier bell nested inside the NUTS curve. If the centers also disagree, or NUTS is visibly bimodal while ADVI is a single spike, you've caught a missed mode — abandon VI for this model.

"But if I have to run NUTS anyway, why bother with VI?" Three honest answers. (1) During *development*, a 5-second ADVI fit lets you iterate on model structure twenty times before you commit to one overnight NUTS run. (2) At *scale*, NUTS on the full dataset may be infeasible, so you validate VI against NUTS on a *subsample*, confirm they agree, then trust VI on the full data. (3) You may only need VI to **initialize** NUTS, in which case "close enough" is the entire requirement.

### Detection method 2: PSIS $\hat{k}$ on the importance weights

The overlay is qualitative and needs a NUTS run. There's a quantitative diagnostic that needs **only the VI fit itself** — it answers "is $q$ a good enough approximation that I can trust importance-weighted corrections of it?" and, by extension, "should I trust $q$ at all?"

> 📜 **Citation/Origin:** Yao, Vehtari, Simpson & Gelman, *"Yes, but Did It Work?: Evaluating Variational Inference,"* ICML 2018 ([arXiv:1802.02538](https://arxiv.org/abs/1802.02538)). This is *the* VI diagnostic paper. Its central tool is the Pareto-$\hat{k}$ from Pareto-smoothed importance sampling (PSIS), repurposed as a VI quality check.

The logic: if $q$ were the true posterior, the **importance weights** $w(\theta) = p(\theta, y) / q(\theta)$ — that is, $\log w = \log p(\theta, y) - \log q(\theta)$ — would be constant. The worse $q$ approximates $p$, the heavier the *right tail* of these weights becomes (a few draws in regions $q$ under-covers get enormous weight). PSIS fits a **generalized Pareto distribution** to that tail and reports its shape parameter $\hat{k}$. A heavy tail $\Rightarrow$ large $\hat{k}$ $\Rightarrow$ $q$ is missing important mass $\Rightarrow$ distrust VI.

The thresholds are the same ones we use for PSIS-LOO in **Chapter 11 (Model Comparison)**, and they'll recur in **Chapter 07 (Diagnostics)**:

| $\hat{k}$ | Verdict for VI |
|---|---|
| $\hat{k} \le 0.5$ | Good. The variational approximation is reliable. |
| $0.5 < \hat{k} \le 0.7$ | OK / borderline. Usable, but treat tail quantities cautiously. |
| $\hat{k} > 0.7$ | **Bad.** $q$ is a poor approximation — do not trust VI; fall back to NUTS. |

Pathfinder computes this $\hat{k}$ for you (it *needs* the importance weights for its PSIS resampling, so the diagnostic is a by-product). Plain ADVI in PyMC does **not** auto-attach a VI $\hat{k}$, so you compute it manually. The recipe per Yao 2018, using ArviZ's PSIS helper:

```python
import numpy as np
import arviz as az

RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)

# 1) Draw from the fitted VI approximation. We work on the model's UNCONSTRAINED space,
#    which is exactly where the Gaussian q lives, so log p and log q are commensurable.
approx = pm.fit(method="advi", random_seed=RANDOM_SEED)   # mean-field ADVI
S = 4000                                                  # number of importance draws
idata_advi = approx.sample(S)

# 2a) log q(θ): the variational log-density. The fitted q is a diagonal Gaussian on the
#     unconstrained space with mean approx.mean.eval() and sd approx.std.eval(); draw the
#     SAME S points from it and score them under that Gaussian (sum over the d coords):
mu_q  = approx.mean.eval()                                # (d,) unconstrained means
sd_q  = approx.std.eval()                                 # (d,) unconstrained sds
Z     = rng.normal(size=(S, mu_q.size))                   # standard-normal noise, reparam trick
theta_u = mu_q + sd_q * Z                                 # (S, d) draws on the UNCONSTRAINED scale
logq  = (-0.5 * ((theta_u - mu_q) / sd_q) ** 2
         - np.log(sd_q) - 0.5 * np.log(2 * np.pi)).sum(axis=1)   # (S,) log N(θ_u; μ_q, σ_q)

# 2b) log p(θ, y) on the SAME unconstrained scale (joint = log-lik + log-prior + the
#     change-of-variables Jacobian). PyMC's compiled logp does all three when called on
#     unconstrained values — that is exactly the function NUTS/ADVI optimize:
logp_fn = approx.model.compile_logp(jacobian=True)        # logp incl. transform Jacobian
point0  = approx.model.initial_point()                    # dict {var_name: array} template
names   = list(point0.keys())                             # unconstrained var order
sizes   = [point0[n].size for n in names]
splits  = np.cumsum(sizes)[:-1]                           # cut points to slice each (d,) row
logp = np.array([
    logp_fn({n: chunk.reshape(point0[n].shape)            # rebuild the {var: array} point dict
             for n, chunk in zip(names, np.split(r, splits))})
    for r in theta_u                                      # one row of (S, d) per draw
])                                                        # (S,)  # verify in current PyMC docs

# 3) Importance log-weights, then PSIS-smooth them and read off k-hat.
#    az.psislw expects log_weights shaped (n_samples,) for the 1-D case (or (chains, draws));
#    here log_weights is the flat (S,) vector built above.
log_weights = logp - logq                                 # (S,)  log w(θ) = log p(θ,y) - log q(θ)
smoothed, k_hat = az.psislw(log_weights)                  # returns smoothed weights and k̂
print(f"VI Pareto k-hat = {float(np.asarray(k_hat)):.2f}")   # > 0.7  ⇒  distrust this VI fit
```

> 🔧 **In practice:** The assembly above is the honest, fully-runnable recipe, but the exact attribute/method names (`approx.model`, `compile_logp(jacobian=...)`, `initial_point()`) do drift between PyMC/`pymc_extras` versions — `# verify in current PyMC docs`. So the *pragmatic* move for plain ADVI is often method 1 (overlay against a short NUTS run), and to use **Pathfinder** when you specifically want the $\hat{k}$ handed to you: because Pathfinder *needs* the importance weights for its own PSIS resampling step, it computes and attaches a Pareto-$\hat{k}$ for free (look for the diagnostics group on the returned `idata` — `# verify the exact diagnostics-group plumbing in your installed pymc_extras`). Treat $\hat{k} > 0.7$ from *any* of these — the manual recipe or Pathfinder's auto-attached one — as a red light.

> 🩺 **Diagnostic — the combined verdict.** Two green lights mean "VI is safe here": (a) the NUTS overlay shows matching centers and only mild interval shrinkage, **and** (b) $\hat{k} \le 0.7$. Either one red — visibly narrower-or-shifted overlay, or $\hat{k} > 0.7$ — means VI is not to be trusted for final inference on this model. Use it only to initialize NUTS, and report the NUTS result.

---

## 9. Worked example: ADVI vs NUTS on a logistic regression

Time to make the lie visible and measurable. We'll fit the **same logistic regression** two ways — mean-field ADVI and NUTS — and overlay the posteriors. I'm using **synthetic data with a known generative process** (the bible's favorite teaching device) so we also get to check whether each method recovers the *true* coefficients. Logistic regression with standardized, roughly-orthogonal predictors is a near-best case for VI, which is exactly the point: even when VI does *well*, you'll still see the characteristic narrowing. If it shrinks here, imagine what it does on a funnel.

### Step 1 — Generate data with known coefficients

```python
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)

# --- known generative process: y ~ Bernoulli(sigmoid(a + X @ beta)) ---
N, D = 800, 3
TRUE_A = -0.4
TRUE_BETA = np.array([1.5, -0.8, 0.6])

X = rng.normal(size=(N, D))                 # already ~standardized (mean 0, sd 1)
eta = TRUE_A + X @ TRUE_BETA
p = 1.0 / (1.0 + np.exp(-eta))              # sigmoid
y = rng.binomial(1, p)                       # observed 0/1 outcomes

print("class balance:", y.mean().round(3))   # ~0.45–0.55, a healthy split
```

We standardized the predictors by construction (the bible's default from Ch 03), which keeps the posterior correlations small and makes mean-field VI's $\sqrt{1-\rho^2}$ penalty mild. Holding onto `TRUE_A` and `TRUE_BETA` lets us grade both methods against ground truth at the end.

### Step 2 — Build the model with named coords

```python
coords = {"pred": ["x1", "x2", "x3"]}

with pm.Model(coords=coords) as logreg:
    X_data = pm.Data("X", X)                              # mutable container (swap-in prediction later)
    # Weakly-informative priors on the STANDARDIZED scale (Ch 03 logic):
    a    = pm.Normal("a", mu=0.0, sigma=1.5)
    beta = pm.Normal("beta", mu=0.0, sigma=1.0, dims="pred")

    eta = a + pm.math.dot(X_data, beta)                  # linear predictor on the log-odds scale
    p   = pm.Deterministic("p", pm.math.sigmoid(eta))    # probability (for PPC / interpretation)

    y_obs = pm.Bernoulli("y_obs", p=p, observed=y)
```

> 🔧 **In practice:** Normal(0, 1) priors on coefficients of *standardized* predictors say "a one-SD change in a predictor probably moves the log-odds by something on the order of $\pm 1$–$2$." That's deliberately weakly-informative — it rules out the absurd (a single standardized predictor flipping the odds by $e^{10}$) without straitjacketing the data. This is the standardize-then-set-priors discipline Chapter 03 established and every later chapter assumes.

### Step 3 — Prior predictive check (a quick sanity pass)

```python
with logreg:
    idata_prior = pm.sample_prior_predictive(draws=500, random_seed=RANDOM_SEED)

# What fraction of "successes" does the prior imply, before seeing data?
prior_p = idata_prior.prior["p"].values.ravel()
print("prior-implied P(y=1): 5th–95th pct =",
      np.percentile(prior_p, [5, 95]).round(2))
```

> 🩺 **Diagnostic:** A well-set logistic prior should spread the prior-predicted success probability across most of $(0,1)$ without piling up at 0 or 1. If your prior implies $P(y{=}1)$ is essentially always 0.999 or 0.001, your priors on the coefficients are too wide for the standardized scale — tighten them. Ours, on Normal priors over standardized $X$, will look sensibly diffuse.

### Step 4 — Fit with NUTS (the reference truth)

```python
with logreg:
    idata_nuts = pm.sample(
        1000, tune=1000, chains=4, target_accept=0.9,
        random_seed=RANDOM_SEED,
    )

az.summary(idata_nuts, var_names=["a", "beta"])
```

> 🩺 **Diagnostic:** Before this counts as "truth," confirm the NUTS run is clean — the full diagnostic loop is Chapter 07, but the headline checks are: **R̂ ≤ 1.01** for every parameter, **ESS_bulk and ESS_tail ≳ 400**, and **0 divergences**. A well-identified logistic regression with 800 rows samples like a dream, so expect R̂ = 1.00 across the board and four-figure ESS. The `mean` column of `az.summary` should land near `TRUE_A = -0.4` and `TRUE_BETA = [1.5, -0.8, 0.6]`, comfortably inside the 94% HDI. That recovery is our anchor.

### Step 5 — Fit the SAME model with mean-field ADVI

```python
with logreg:
    approx = pm.fit(n=30_000, method="advi", random_seed=RANDOM_SEED)

idata_advi = approx.sample(4000)             # InferenceData, drawn from the fitted Gaussian q

# Convergence of the optimization (NOT of the approximation quality!):
plt.figure(figsize=(7, 3))
plt.plot(approx.hist)
plt.xlabel("ADVI iteration"); plt.ylabel("negative ELBO (loss)")
plt.title("ELBO trajectory — should drop then flatten into a noisy plateau")
```

> 🩺 **Diagnostic — `approx.hist`:** You'll see a steep initial drop (the optimizer racing toward the mode) settling into a flat, jittery plateau over the last several thousand iterations. Flat = the *optimization* converged. Remember the refrain: that says nothing yet about whether $q$ matches $p$. For that we overlay.

### Step 6 — Overlay the posteriors and MEASURE the underestimation

```python
# Side-by-side intervals — the lie made visible.
az.plot_forest(
    [idata_nuts, idata_advi],
    model_names=["NUTS", "ADVI"],
    var_names=["a", "beta"],
    combined=True, figsize=(8, 5),
)
plt.axvline(0, color="k", ls=":", alpha=0.4)

# Now MEASURE it: ratio of posterior SDs (ADVI / NUTS) per parameter.
sd_nuts = az.summary(idata_nuts, var_names=["a", "beta"])["sd"]
sd_advi = az.summary(idata_advi, var_names=["a", "beta"])["sd"]
ratio = (sd_advi / sd_nuts).round(3)
print("posterior SD ratio  ADVI / NUTS  (≈1 good, <1 = ADVI too narrow):")
print(ratio)
```

> 🩺 **Diagnostic — reading the overlay.** On the forest plot, each parameter shows two stacked intervals. The **centers** (posterior means) will nearly coincide — VI gets the *location* right here. The **ADVI intervals will be visibly shorter**, nested inside the NUTS intervals. That gap is the variance underestimation you derived in §2, now staring at you from a plot. The printed **SD ratio** quantifies it: on this well-conditioned, standardized logistic regression expect ratios in the low-to-mid 0.9s — ADVI runs maybe 5–15% too narrow. Mild, because the predictors are near-orthogonal. Crank up the predictor correlation (make the columns of `X` collinear) and re-run: the ratios plunge, exactly as the $\sqrt{1-\rho^2}$ argument predicts. *That* is the experiment that makes the theory unforgettable.

### Step 7 — Confirm both recovered the truth, then quantify with $\hat{k}$

```python
# Did each method recover the true generative parameters?
truth = {"a": TRUE_A, "x1": 1.5, "x2": -0.8, "x3": 0.6}
print("\nNUTS means:", az.summary(idata_nuts, var_names=["a","beta"])["mean"].round(2).to_dict())
print("ADVI means:", az.summary(idata_advi, var_names=["a","beta"])["mean"].round(2).to_dict())
print("Truth:     ", truth)
```

Both methods land their posterior means close to the truth — VI's *point* estimates are fine here; it's the *uncertainty* that's understated. As the §8 quantitative check, you'd compute the VI Pareto-$\hat{k}$ on $\log p - \log q$ (the manual `az.psislw` recipe from §8); for a clean logistic regression like this you'd expect $\hat{k} \le 0.5$ — a green light confirming ADVI is genuinely trustworthy here, modulo the mild narrowing. Now hold onto that green light, because **Step 9 below runs the exact opposite case** — Eight Schools, where the same machinery sends $\hat{k}$ past 0.7 and the overlay falls apart.

> 💡 **Why the funnel breaks VI.** The case we are about to fit in Step 9 is the **Eight Schools** hierarchical model in its non-centered form ($z_j \sim \mathrm{Normal}(0,1)$, $\theta_j = \mu + \tau z_j$). It falls apart at the *funnel*: the posterior for the group scale $\tau$ has a sharp, skewed neck near zero that no Gaussian $q$ can match. Mean-field ADVI reports a $\tau$ far too small and far too narrow; NUTS (with its non-centered parameterization, the cure we develop fully in **Chapter 10**) shows the true heavy lower tail; the overlay diverges dramatically; and the VI $\hat{k}$ exceeds 0.7. **Pathfinder** does better than plain ADVI on the funnel but still can't fully resolve the neck — which is precisely why it's a great *initializer* for NUTS rather than a replacement. That contrast — VI succeeding on the GLM, failing on the funnel — is the whole moral of the chapter in one experiment, and we now run it.

### Step 8 — Posterior predictive check (works identically for either fit)

```python
with logreg:
    # extend_inferencedata=True mutates idata_advi IN PLACE (adds the posterior_predictive
    # group) and returns the same object, so no reassignment is needed here.
    pm.sample_posterior_predictive(
        idata_advi, extend_inferencedata=True, random_seed=RANDOM_SEED,
    )

az.plot_ppc(idata_advi, num_pp_samples=100)   # replicated y vs observed y
```

> 🩺 **Diagnostic:** For a binary outcome the PPC overlays the distribution of replicated success-counts against the observed count. ADVI's predicted-success distribution should bracket the observed value here — because VI nailed the *means*, its *predictions* look fine even though its *parameter uncertainty* is understated. This is a subtle trap: **good posterior predictive checks do not rescue understated parameter uncertainty.** A model can predict the training outcomes well and still report overconfident coefficient intervals. Always check the parameter-level overlay (Step 6), not just the PPC.

### Step 9 — Now SHOW the failure: Eight Schools and the funnel

The GLM above is VI's best case, and even there it ran narrow. To complete the moral of the chapter we have to *show* the failure, not just assert it — so here is the contrast in code: the **Eight Schools** hierarchical model (the named dataset from §7) in its **non-centered** parameterization, fit three ways. This is the funnel: the group-scale $\tau$ and the standardized offsets $z_j$ are entangled, and as $\tau \to 0$ the posterior pinches into a sharp neck that *no* Gaussian $q$ can represent.

```python
# Eight Schools — the canonical funnel (data from §7).
y_es     = np.array([28.,  8., -3.,  7., -1.,  1., 18., 12.])
sigma_es = np.array([15., 10., 16., 11.,  9., 11., 10., 18.])

with pm.Model(coords={"school": np.arange(8)}) as eight_schools:
    mu    = pm.Normal("mu", 0.0, 10.0)
    tau   = pm.HalfNormal("tau", 10.0)                  # group scale — the funnel's culprit
    z     = pm.Normal("z", 0.0, 1.0, dims="school")     # NON-CENTERED offsets
    theta = pm.Deterministic("theta", mu + tau * z, dims="school")
    pm.Normal("y_obs", mu=theta, sigma=sigma_es, observed=y_es, dims="school")

    # 1) NUTS — the reference truth (raise target_accept; funnels breed divergences)
    idata_nuts_es = pm.sample(1000, tune=1000, chains=4, target_accept=0.95,
                              random_seed=RANDOM_SEED)
    # 2) mean-field ADVI
    approx_es     = pm.fit(n=40_000, method="advi", random_seed=RANDOM_SEED)
    idata_advi_es = approx_es.sample(4000)

# 3) Pathfinder (pymc_extras) — better than ADVI on the funnel, still not a full cure
import pymc_extras as pmx
with eight_schools:
    idata_path_es = pmx.fit(method="pathfinder", num_paths=8, num_draws=4000, jitter=12,
                            random_seed=RANDOM_SEED)   # verify in current pymc_extras docs

# Overlay tau (the scale that VI cannot get right) and mu across all three:
az.plot_forest([idata_nuts_es, idata_advi_es, idata_path_es],
               model_names=["NUTS", "ADVI", "Pathfinder"],
               var_names=["mu", "tau"], combined=True, figsize=(8, 4))

print("tau posterior summary (watch the sd and the lower tail):")
for name, idata in [("NUTS", idata_nuts_es), ("ADVI", idata_advi_es),
                    ("Pathfinder", idata_path_es)]:
    s = az.summary(idata, var_names=["tau"])
    print(f"  {name:11s}  mean={float(s['mean'].iloc[0]):.2f}  "
          f"sd={float(s['sd'].iloc[0]):.2f}")
```

> 🩺 **Diagnostic — reading the funnel forest plot.** On the $\tau$ row you will see the lie in full color. **NUTS** shows a broad, right-skewed interval whose lower edge crowds toward zero — the true heavy lower tail of the funnel. **Mean-field ADVI** reports a $\tau$ interval that is dramatically *narrower* and *shifted up* off the neck: it cannot place mass near $\tau \approx 0$ because the non-centered geometry there is a pinched, non-Gaussian wedge, so it parks $q$ in the fat part of the funnel and reports false confidence in a moderate $\tau$. **Pathfinder** lands *between* the two — visibly closer to NUTS than ADVI is (its along-the-path Gaussian and PSIS reweighting recover some of the spread), but still failing to fully resolve the neck. If you assemble the VI $\hat{k}$ from §8's recipe on `approx_es`, it will sail past **0.7** — the quantitative confirmation that this $q$ is not to be trusted. That single plot — *VI succeeding on the GLM (Step 6), failing on the funnel here* — is the entire chapter in one experiment. Exercise 5 asks you to extend it and quantify each method's $\tau$ underestimation.

---

## 10. When to actually reach for VI

You now understand VI well enough to use it on purpose. Here is the decision guide I'd give a teammate.

**Reach for VI when:**

- **The data is massive.** When a single full-data NUTS gradient is too expensive to evaluate thousands of times, **minibatch ADVI** — which subsamples the data each step — may be your only Bayesian option. The mechanics (note the *single coordinated* `pm.Minibatch` call — this is the part people get wrong):
  ```python
  # X is (N, D), y is (N,). ONE Minibatch call slices BOTH with the SAME row index,
  # so predictor rows and outcome rows stay paired every step:
  X_mb, y_mb = pm.Minibatch(X, y, batch_size=500)      # shared random subsample each step
  with pm.Model() as big_model:
      a    = pm.Normal("a", 0.0, 1.5)
      beta = pm.Normal("beta", 0.0, 1.0, shape=X.shape[1])
      eta  = a + pm.math.dot(X_mb, beta)               # linear predictor built from the SAME batch
      p    = pm.math.sigmoid(eta)
      pm.Bernoulli("y_obs", p=p, observed=y_mb,
                   total_size=N)                        # total_size IS REQUIRED — rescales the likelihood
      approx = pm.fit(method="advi", random_seed=RANDOM_SEED)
  ```
  Two non-negotiables hide here. **(1) Pair the tensors in one `pm.Minibatch(X, y, ...)` call.** Two *separate* `pm.Minibatch(X, ...)` and `pm.Minibatch(y, ...)` calls each draw their *own* random row index every step, so `X_mb` and `y_mb` get sliced with *different* rows — the model silently trains on mismatched $(x_i, y_j)$ pairs and the fit is garbage with no error raised. The coordinated single call shares one index across both tensors. **(2) `total_size=N` tells PyMC the minibatch is a sample of a size-$N$ dataset so it rescales the log-likelihood correctly. Omit it and your posterior scale is silently wrong.** We return to minibatch ADVI in **Chapter 17 (Scaling)**.
- **You're iterating fast on model structure.** A 5-second ADVI fit lets you try twenty model variants before committing one to an overnight NUTS run. Use VI to *develop*, NUTS to *confirm*.
- **You need to initialize MCMC.** Pathfinder (or ADVI) gives NUTS a good starting point and mass matrix, shrinking warmup on hard models. This is the most robust use of VI in a serious workflow — you never have to *trust* the approximation, only point with it.
- **The posterior is well-conditioned and you've validated it.** Big-$n$, well-identified GLMs with standardized predictors (like §9) have near-Gaussian, weakly-correlated posteriors where mean-field VI is genuinely close. If the NUTS overlay and $\hat{k}$ both come up green on a representative subsample, you can trust VI on the full data.

**Do NOT reach for VI (use NUTS) when:**

- The posterior is **multimodal** — mode-seeking will silently drop modes.
- There's a **funnel or strong hierarchy** — the neck defeats every Gaussian $q$ (Eight Schools, Ch 10).
- You need **calibrated tail probabilities or honest uncertainty for decisions** — underdispersion makes VI intervals untrustworthy precisely where the stakes are highest.
- It's the **final, reported inference** and you haven't validated VI against NUTS — when in doubt, the gold standard is one `pm.sample()` away.

> 💡 **The one-sentence policy:** Use VI to go *fast* and to *initialize*; use NUTS to be *right*; and never report a VI posterior you haven't checked against NUTS (overlay) or $\hat{k}$.

---

## ⚠️ Common errors & how to fix them

| Symptom | Cause | Fix |
|---|---|---|
| `pm.quap` / `pm.Laplace(...)` "not found" for VI | Confusing names: `quap` is a `rethinking` R concept; `pm.Laplace` is the *double-exponential distribution*, not the Laplace approximation | For VI use `pm.fit(method="advi"/"fullrank_advi")`; for the Laplace *approximation* see Ch 04 (`pymc_extras.fit_laplace`) |
| ADVI ELBO (`approx.hist`) noisy / never flattens | Learning rate too high, too few iterations, or a multimodal target | Raise `n`; lower the optimizer LR; add `CheckParametersConvergence`; try `fullrank_advi`; if multimodal, switch to NUTS |
| ELBO diverges to $\pm\infty$ in the first iterations | Unstandardized data or extreme priors blow up the initial gradient | **Standardize predictors** (Ch 03); set weakly-informative priors; try `obj_optimizer=pm.adagrad_window(...)` |
| VI intervals far narrower than NUTS | Reverse-KL variance underestimation — **inherent, not a bug** | Expected; *report* it. Use `fullrank_advi` or Pathfinder; standardize to decorrelate; for the final answer use NUTS |
| VI misses posterior correlations / the funnel | Mean-field's diagonal $q$ can't represent correlation; no Gaussian fits a funnel neck | `fullrank_advi` for correlations; **non-centered** reparameterization (Ch 10) for funnels; ultimately NUTS |
| VI reports a single crisp mode; you suspect more | Mode-seeking reverse KL collapses onto one mode | Multi-path Pathfinder (raise `num_paths`, `jitter`); SVGD; or NUTS, which explores all modes if mixing is good |
| Minibatch ADVI gives a wrongly-scaled posterior | Missing `total_size` on the observed node | Always pass `total_size=N` so the subsampled likelihood is rescaled to full-data size |
| Pathfinder gives a confident wrong answer | All paths land in one bad mode; or the Gaussian family can't fit the posterior | Raise `num_paths` and `jitter`; keep `importance_sampling="psis"`; verify against NUTS |
| Pathfinder shows high Pareto-$\hat{k}$ everywhere | `jacobian_correction=False`, or a genuinely poor Gaussian fit | Keep `jacobian_correction=True`; if still high, the Gaussian family is inadequate → NUTS |
| Can't compute LOO / model comparison after VI | No pointwise log-likelihood stored, and LOO assumes posterior samples | For the NUTS reference, sample with `idata_kwargs={"log_likelihood": True}`; for VI use the $\hat{k}$-from-PSIS check (Yao 2018), not LOO |
| `az.psislw` gives nonsense $\hat{k}$ for VI | `logp` and `logq` computed on *different* (constrained vs unconstrained) scales | Compute both on the **same unconstrained scale** the variational $q$ lives on; `# verify in current PyMC docs` |

---

## 🧪 Exercises

1. **(Conceptual — the sign that changes everything.)** Write out $\mathrm{KL}(q\|p)$ and $\mathrm{KL}(p\|q)$ for a *bimodal* target $p$ (two well-separated Gaussian bumps) approximated by a *single* Gaussian $q$. Argue from the integrands which divergence makes $q$ pick one bump (mode-seeking) and which makes $q$ straddle both, planting mass in the empty valley between (mass-covering). State which one VI uses and why we're stuck with it. *Hint: focus on regions where one density is near zero while the other is not.*

2. **(Derivation.)** Starting from $\log p(y) = \mathbb{E}_q[\log p(y)]$, reproduce the ELBO/KL decomposition $\log p(y) = \mathrm{ELBO}(q) + \mathrm{KL}(q\|p)$ in full, then derive *both* alternative forms of the ELBO: the "expected log-likelihood minus prior-KL" form and the "expected log-joint plus entropy" form. For each, write one sentence on what intuition it provides. *Hint: $p(\theta,y) = p(y\mid\theta)p(\theta)$ for the first; $\mathbb{H}[q] = -\mathbb{E}_q[\log q]$ for the second.*

3. **(Coding — make underestimation a dial.)** Take the §9 logistic regression and add a tuning knob: generate the predictor matrix `X` so that columns 1 and 2 have correlation $\rho \in \{0, 0.5, 0.9, 0.99\}$ (use a Cholesky factor of the target correlation matrix). For each $\rho$, fit both mean-field ADVI and NUTS and record the SD ratio `sd_advi / sd_nuts` for `beta`. Plot the ratio against $\rho$ and overlay the theoretical $\sqrt{1-\rho^2}$ curve. *Hint: the mean-field underestimation factor for a correlated Gaussian pair is exactly $\sqrt{1-\rho^2}$ — your empirical points should track it.*

4. **(Coding — full-rank to the rescue?)** Re-run Exercise 3 at $\rho = 0.9$ but with `method="fullrank_advi"`. Does full-rank recover the NUTS interval widths? Then make the posterior *non-Gaussian*: switch the coefficient priors to heavy-tailed `pm.StudentT(nu=3, ...)` and a smaller $N$. Does full-rank still match NUTS? Explain what you observe in terms of "full-rank fixes correlation but not non-Gaussianity."

5. **(Coding — VI breaks on the funnel.)** Fit **Eight Schools** (`y=[28,8,-3,7,-1,1,18,12]`, `sigma=[15,10,16,11,9,11,10,18]`) in the **non-centered** parameterization ($z_j \sim \mathrm{Normal}(0,1)$, $\theta_j = \mu + \tau z_j$, with $\tau \sim \mathrm{HalfNormal}$). Fit it three ways — mean-field ADVI, Pathfinder, and NUTS — and `az.plot_forest` all three for $\tau$ and $\mu$. Which method most underestimates $\tau$? Where does Pathfinder sit relative to ADVI and NUTS? *Hint: focus on the lower tail of $\tau$ — VI cannot represent the funnel's narrow neck near zero.*

6. **(Diagnostic — build the $\hat{k}$ check.)** For the §9 ADVI fit, assemble the importance log-weights $\log p(\theta,y) - \log q(\theta)$ on the unconstrained scale, run `az.psislw`, and report $\hat{k}$. Confirm it lands $\le 0.5$ (good) for the clean GLM. Then sabotage the model — shrink $N$ to 30 and widen the priors — refit ADVI, recompute $\hat{k}$, and watch it climb. Relate the $\hat{k}$ value to the size of the NUTS-vs-ADVI overlay gap. *Hint: a heavier importance-weight tail (bigger $\hat{k}$) and a bigger overlay gap are two views of the same approximation failure.*

---

## 📚 Resources & further reading

**Foundational papers**

- **Blei, Kucukelbir & McAuliffe (2017), "Variational Inference: A Review for Statisticians,"** JASA — [arXiv:1601.00670](https://arxiv.org/abs/1601.00670). The single best conceptual reference for this chapter: the ELBO/CAVI derivation, mean-field, and the variance-underestimation discussion, all rigorous and readable.
- **Kucukelbir, Tran, Ranganath, Gelman & Blei (2017), "Automatic Differentiation Variational Inference,"** *JMLR* 18(14):1–45 — [jmlr.org/papers/v18/16-107.html](https://jmlr.org/papers/v18/16-107.html) ([arXiv:1603.00788](https://arxiv.org/abs/1603.00788)). The ADVI paper: transform → Gaussian → SGD ELBO, and the mean-field vs full-rank section is required reading.
- **Zhang, Carpenter, Gelman & Vehtari (2022), "Pathfinder: Parallel quasi-Newton variational inference,"** *JMLR* 23(306):1–49 — [jmlr.org/papers/v23/21-0889.html](https://jmlr.org/papers/v23/21-0889.html) ([arXiv:2108.03782](https://arxiv.org/abs/2108.03782)). The Pathfinder paper; see the figures on KL-along-the-path and multi-path PSIR.
- **Yao, Vehtari, Simpson & Gelman (2018), "Yes, but Did It Work?: Evaluating Variational Inference,"** ICML — [arXiv:1802.02538](https://arxiv.org/abs/1802.02538) ([PMLR PDF](http://proceedings.mlr.press/v80/yao18a/yao18a.pdf)). *The* VI diagnostics paper: PSIS $\hat{k}$ for VI ($\hat{k} > 0.7 \Rightarrow$ distrust) and VSBC. Use its $\hat{k}$ thresholds.

**Books**

- **Martin, Kumar & Lao, *Bayesian Modeling and Computation in Python*** — free at [bayesiancomputationbook.com](https://bayesiancomputationbook.com). **Chapter 11, "Appendiceal Topics,"** covers Laplace, variational inference, grid approximation, and other inference internals with idiomatic PyMC/ArviZ — the closest companion to this chapter.
- **McElreath, *Statistical Rethinking* (2nd ed., 2020).** The `quap` / quadratic-approximation thread (SR Ch 2 & 4) is the Laplace cousin of VI; useful for the §5 transform-to-unconstrained intuition. PyMC port in [pymc-devs/pymc-resources](https://github.com/pymc-devs/pymc-resources).
- **Gelman et al., *Bayesian Data Analysis* (BDA3, 2013).** The reference; see the modal/normal-approximation and the advanced computation chapters for the theory behind Laplace and VI.

**Docs & runnable examples (primary code sources)**

- **PyMC Variational API quickstart** — [pymc.io/projects/examples/.../variational_api_quickstart.html](https://www.pymc.io/projects/examples/en/latest/variational_inference/variational_api_quickstart.html). The canonical `pm.fit` / `approx.sample` / `approx.hist` / `Tracker` / minibatch reference — the primary code source for this chapter.
- **`pymc.fit` API** — [pymc.io/projects/docs/.../pymc.fit.html](https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.fit.html). Confirms the `method` strings: `advi`, `fullrank_advi`, `svgd`, `asvgd`.
- **`pymc.ADVI`** — [pymc.io/projects/docs/.../pymc.ADVI.html](https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.ADVI.html) and **`pymc.FullRankADVI`** — [pymc.io/projects/docs/.../pymc.FullRankADVI.html](https://www.pymc.io/projects/docs/en/latest/api/generated/pymc.FullRankADVI.html). The OO interface: `.fit`, `.refine`, mean-field vs full-rank.
- **PyMC Pathfinder example** — [pymc.io/projects/examples/.../pathfinder.html](https://www.pymc.io/projects/examples/en/latest/variational_inference/pathfinder.html). Runnable `pmx.fit(method="pathfinder", ...)` on Eight Schools with `az.plot_forest` vs NUTS — copy this idiom for Exercise 5.
- **`pymc_extras.fit_pathfinder` API** — [pymc.io/projects/extras/.../fit_pathfinder.html](https://www.pymc.io/projects/extras/en/stable/generated/pymc_extras.inference.fit_pathfinder.html). Full signature (`num_paths`, `importance_sampling`, `inference_backend`, `jitter`, `jacobian_correction`).
- **NumPyro AutoGuide docs** — [num.pyro.ai/en/stable/autoguide.html](https://num.pyro.ai/en/stable/autoguide.html). `AutoNormal` / `AutoMultivariateNormal` / `AutoLaplaceApproximation` / `AutoIAFNormal` / `AutoBNAFNormal` — the JAX VI cross-reference for Ch 17.
- **Stan Reference Manual — Pathfinder** — [mc-stan.org/docs/reference-manual/pathfinder.html](https://mc-stan.org/docs/reference-manual/pathfinder.html). Stan's authoritative Pathfinder description, for the CmdStanPy cross-reference.
- **ArviZ `summary`** — [python.arviz.org/.../arviz.summary.html](https://python.arviz.org/en/stable/api/generated/arviz.summary.html). Works on VI / Pathfinder InferenceData; Pareto-$\hat{k}$ bands ($\le 0.5$ good, $\le 0.7$ ok, $> 0.7$ bad).
- **PyMC Discourse: "Laplace approximation vs VI"** — [discourse.pymc.io/t/laplace-approximation-vs-vi/17722](https://discourse.pymc.io/t/laplace-approximation-vs-vi/17722). Practitioner guidance on when to reach for each.

---

## ➡️ What's next

You can now turn inference into optimization, you know exactly why the reverse-KL bargain costs you variance, and — most importantly — you can *catch* VI when it lies by overlaying it against NUTS and reading the Pareto-$\hat{k}$. But notice how often this chapter leaned on a phrase like "confirm the NUTS run is clean first." That confirmation is a discipline of its own, and it's where we go next: **Chapter 07 — MCMC Diagnostics and Debugging.** There we make the headline checks rigorous — rank-normalized R̂, ESS (bulk and tail), MCSE, trace and rank and forest plots, divergences, the energy/BFMI diagnostic — and we confront the funnel head-on, developing the **non-centered parameterization** that this chapter kept promising as the cure. Once you can *trust* a NUTS run, you'll have the gold-standard reference against which every approximation in this course — Laplace, ADVI, Pathfinder — is ultimately judged.
