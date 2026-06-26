# Chapter 05 — Inference II: MCMC from Metropolis to NUTS

### **How a Markov chain learns to walk the typical set — from the 1953 Metropolis algorithm to the gradient-powered sampler that runs inside `pm.sample()`**

> *A note from me to you.*
>
> *In the last chapter we computed posteriors three ways — analytically (conjugate algebra), on a grid, and with a quadratic (Laplace) approximation. Each of those has a hard ceiling. Conjugacy demands a lucky prior–likelihood pair. The grid drowns the moment you have more than two or three parameters — a 100-point grid in 10 dimensions is $10^{20}$ evaluations, more than the number of grains of sand on Earth, every time you want one posterior. And the quadratic approximation lies to you the instant your posterior is skewed, heavy-tailed, multimodal, or funnel-shaped — which is to say, the instant you build a model worth building.*
>
> *This chapter is where Bayesian inference becomes practical for real models. The trick, which felt like sorcery to me the first time I understood it, is this: you can compute expectations under a distribution you cannot integrate, cannot normalize, and cannot even visualize — as long as you can write down its **unnormalized** density and (for the good samplers) its **gradient**. Markov chain Monte Carlo is the machine that does this.*
>
> *I want you to leave this chapter with two things. First, a from-scratch, in-your-bones understanding of why the obvious idea — random-walk Metropolis — works in two dimensions and then catastrophically fails in two hundred. That failure is not a bug; it is geometry, and once you see it you will never un-see it. Second, a precise mental model of what Hamiltonian Monte Carlo and NUTS actually do when you type `pm.sample()` — why momentum, why gradients, why leapfrog, what `target_accept` adjusts, what the mass matrix is, and what a divergence is screaming at you. Almost every painful PyMC debugging session you will ever have is a conversation with the algorithm in this chapter. If you understand the algorithm, you can hear what it's saying.*
>
> *We will write a Metropolis sampler by hand in plain NumPy — no PyMC magic — so there is nothing hidden. Then we will watch it die in high dimensions. Then we will earn HMC as the cure. Let's go.*

---

## What you'll be able to do after this chapter

- Explain what a **Markov chain** is, why a well-designed one has the posterior as its **stationary distribution**, and what **detailed balance** guarantees.
- Derive and **code from scratch** a random-walk **Metropolis** sampler in NumPy, and explain why it only needs the *unnormalized* posterior (the evidence cancels).
- Describe **Gibbs sampling** as a special case, why its acceptance is always 1, and why it **zig-zags** to a crawl under posterior correlation.
- Articulate Betancourt's **typical set** picture: why posterior mass lives on a thin shell *away* from the mode, and why **random-walk MCMC collapses** as dimension grows — and demonstrate that collapse numerically with effective sample size.
- Explain **Hamiltonian Monte Carlo** end to end: momentum augmentation, the Hamiltonian $H = U + K$, Hamilton's equations, the **leapfrog** integrator (kick–drift–kick, why it's symplectic), momentum refresh, and the Metropolis correction for discretization error.
- Explain **NUTS**: recursive doubling, the **no-U-turn** termination criterion, `max_treedepth`, and why modern implementations use multinomial sampling.
- Explain **warmup/adaptation**: dual-averaging step size, the **mass matrix** (diagonal vs dense), and exactly what raising `target_accept` does.
- Decode **every knob** of `pm.sample()` and know the difference between the library default `target_accept=0.8` and this course's house default `0.9`.

---

## 0. Setup and the one idea everything rests on

```python
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
import scipy.stats as stats

RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)
```

Here is the single problem MCMC solves. Bayesian inference is, in the end, almost never about the posterior *density* — it's about **expectations** under the posterior:

$$
\mathbb{E}_{p(\theta\mid y)}[f(\theta)] = \int f(\theta)\, p(\theta\mid y)\, d\theta.
$$

A posterior mean is $f(\theta)=\theta$. A posterior probability that a slope is positive is $f(\theta)=\mathbb{1}[\theta>0]$. A predictive distribution is an expectation of the likelihood over the posterior. **Everything you report is an integral against the posterior.** And those integrals are, in general, analytically hopeless and (past a couple of dimensions) numerically hopeless by brute force.

Monte Carlo replaces the integral with an average over samples:

$$
\mathbb{E}_{p}[f] \approx \frac{1}{N}\sum_{n=1}^{N} f(\theta_n), \qquad \theta_n \sim p(\theta\mid y).
$$

If you could draw independent samples from the posterior, you'd be done — the law of large numbers guarantees convergence, and the error shrinks like $1/\sqrt{N}$ *regardless of dimension* (this dimension-independence is the deep reason Monte Carlo beats grids). The catch: you **cannot** draw independent samples from a posterior you only know up to a constant. MCMC's bargain is to give up independence. Instead of i.i.d. draws, it produces a **dependent sequence** — a Markov chain — engineered so that, after it forgets where it started, its samples have the posterior as their long-run distribution. The averages still converge; you just need more samples because consecutive ones are correlated. Quantifying *how many more* is the job of **effective sample size**, which will become our scoreboard.

> 💡 **Intuition:** Think of the posterior as a landscape of "where the believable parameter values are." MCMC sends a hiker wandering that landscape under rules rigged so that, over a long enough walk, the *fraction of time* the hiker spends in any region equals that region's posterior probability. Count how often the hiker visits a place; that's your estimate of how probable it is.

---

## 1. Markov chains, stationarity, and detailed balance

A **Markov chain** is a sequence $\theta_0, \theta_1, \theta_2, \dots$ where the next state depends only on the current one — not on the full history. The rule for moving is a **transition kernel** $T(\theta'\mid\theta)$: the probability density of jumping to $\theta'$ given you're at $\theta$. "Markov" means memorylessness:

$$
p(\theta_{n+1}\mid \theta_n, \theta_{n-1}, \dots, \theta_0) = p(\theta_{n+1}\mid \theta_n) = T(\theta_{n+1}\mid\theta_n).
$$

We want to design $T$ so that the chain's long-run distribution is exactly our posterior $p(\theta\mid y)$. The technical name for "long-run distribution" is the **stationary** (or **invariant**) distribution: a distribution that, once the chain reaches it, it stays in. Formally, $p$ is stationary for $T$ if pushing $p$ through one step of the kernel returns $p$:

$$
\int T(\theta'\mid\theta)\, p(\theta)\, d\theta = p(\theta').
$$

In words: if right now your position is distributed according to $p$, then after one transition it's *still* distributed according to $p$. The distribution is a fixed point of the dynamics.

Stationarity alone isn't enough. We also need:

- **Irreducibility** — from any starting point the chain can eventually reach any region of positive posterior probability. (No part of the space is walled off.)
- **Aperiodicity** — the chain doesn't get trapped in a deterministic cycle that returns to states on a fixed clock.

With those plus a stationary distribution, the **ergodic theorem** delivers the payoff we wanted:

$$
\frac{1}{N}\sum_{n=1}^{N} f(\theta_n)\ \xrightarrow{\ N\to\infty\ }\ \mathbb{E}_{p}[f],
$$

*for almost any starting point* and *even though the samples are correlated*. This is the theorem that licenses everything we do. It says: build an irreducible, aperiodic chain with the posterior as its stationary distribution, run it long enough, and time-averages converge to posterior expectations.

> 🧮 **The math:** The "run it long enough" hides a subtlety. Early samples reflect the (arbitrary) starting point, not the posterior. That transient is **warmup/burn-in**, and we throw it away. What remains still isn't independent — neighbors are correlated — so the *effective* number of independent samples is smaller than $N$. We'll make this precise with ESS in §3 and again, exhaustively, in **Chapter 07 (MCMC Diagnostics and Debugging)**.

### Detailed balance: an easy way to guarantee stationarity

Designing a $T$ whose stationary distribution is a *specified* $p$ sounds hard. There's a beautiful sufficient condition that makes it almost mechanical: **detailed balance** (also called **reversibility**). A kernel satisfies detailed balance with respect to $p$ if

$$
p(\theta)\, T(\theta'\mid\theta) = p(\theta')\, T(\theta\mid\theta') \qquad \text{for all } \theta, \theta'.
$$

```
   p(θ) · T(θ'|θ)   =   p(θ') · T(θ|θ')
   ┕━ rate of flow ━┙       ┕━ rate of flow ━┙
      θ  →  θ'                 θ' →  θ
```

Read it as a flow-balance statement: in equilibrium, the *probability flux* from $\theta$ to $\theta'$ exactly equals the flux back from $\theta'$ to $\theta$. Every pairwise exchange is balanced. If that holds, integrate both sides over $\theta$:

$$
\int p(\theta)\,T(\theta'\mid\theta)\,d\theta = \int p(\theta')\,T(\theta\mid\theta')\,d\theta = p(\theta')\underbrace{\int T(\theta\mid\theta')\,d\theta}_{=\,1} = p(\theta').
$$

That's *exactly* the stationarity condition. So **detailed balance $\Rightarrow$ stationarity**. We've turned a global integral condition into a local, checkable, pairwise condition. This is the lever Metropolis–Hastings pulls.

> ⚠️ **Pitfall:** Detailed balance is **sufficient but not necessary**. A chain can be stationary for $p$ without being reversible. This matters because **HMC is not reversible in the naïve sense** — it uses a more subtle argument (volume preservation + a momentum flip that makes the *composite* move reversible). Don't go away thinking "all MCMC satisfies detailed balance"; Metropolis and Gibbs do, HMC earns its correctness differently. We'll flag exactly where.

---

## 2. Metropolis–Hastings: the algorithm that started it all

Now we build a concrete reversible chain. The **Metropolis–Hastings** (MH) recipe is two steps repeated forever:

1. **Propose.** From the current state $\theta$, draw a candidate $\theta' \sim q(\theta'\mid\theta)$ from a **proposal distribution** $q$ you choose (e.g., a Gaussian centered at $\theta$).
2. **Accept or reject.** Compute the acceptance probability
$$
\alpha(\theta\to\theta') = \min\!\left(1,\ \frac{p(\theta')\, q(\theta\mid\theta')}{p(\theta)\, q(\theta'\mid\theta)}\right),
$$
draw $u\sim\mathrm{Uniform}(0,1)$, and **move** to $\theta'$ if $u<\alpha$; otherwise **stay** at $\theta$ (the current state is repeated in the chain).

That ratio is engineered — it's not arbitrary — so that the resulting kernel satisfies detailed balance for $p$. Let me show you that it does, because the proof is short and it makes the magic concrete.

> 🧮 **The math — why this exact $\alpha$ gives detailed balance.** For $\theta\neq\theta'$, the transition density is "propose then accept": $T(\theta'\mid\theta) = q(\theta'\mid\theta)\,\alpha(\theta\to\theta')$. Plug into the detailed-balance product and suppose WLOG that the ratio $R = \frac{p(\theta')q(\theta\mid\theta')}{p(\theta)q(\theta'\mid\theta)} \le 1$, so $\alpha(\theta\to\theta')=R$ and the reverse $\alpha(\theta'\to\theta)=\min(1, 1/R)=1$. Then
> $$ p(\theta)\,T(\theta'\mid\theta) = p(\theta)\,q(\theta'\mid\theta)\cdot \frac{p(\theta')q(\theta\mid\theta')}{p(\theta)q(\theta'\mid\theta)} = p(\theta')\,q(\theta\mid\theta'), $$
> $$ p(\theta')\,T(\theta\mid\theta') = p(\theta')\,q(\theta\mid\theta')\cdot 1 = p(\theta')\,q(\theta\mid\theta'). $$
> The two sides are equal. Detailed balance holds, so $p$ is stationary. The $\min(1,\cdot)$ is precisely what's needed to make the forward and backward fluxes match.

### The single most important consequence: you only need the *unnormalized* posterior

Look hard at the ratio $\frac{p(\theta')}{p(\theta)}$ in the Metropolis case (symmetric proposal — more on that in a second). The posterior is

$$
p(\theta\mid y) = \frac{p(y\mid\theta)\,p(\theta)}{p(y)},
$$

and the denominator $p(y)$ — the **evidence/marginal likelihood**, the thing we said in **Chapter 01 (Foundations)** is the genuinely hard integral — is a *constant in $\theta$*. So in the ratio it cancels exactly:

$$
\frac{p(\theta'\mid y)}{p(\theta\mid y)} = \frac{p(y\mid\theta')\,p(\theta')\,/\,p(y)}{p(y\mid\theta)\,p(\theta)\,/\,p(y)} = \frac{p(y\mid\theta')\,p(\theta')}{p(y\mid\theta)\,p(\theta)}.
$$

```
       p(θ'|y)      likelihood(θ')·prior(θ')      [ p(y) cancels ]
  α ∝ ─────────  =  ───────────────────────────
       p(θ |y)      likelihood(θ )·prior(θ )
```

**This is why MCMC is possible at all.** We never compute $p(y)$. We only ever need the *unnormalized* posterior $\tilde p(\theta) = p(y\mid\theta)\,p(\theta)$ — likelihood times prior, both of which we can always evaluate. The intractable normalizing constant that blocks analytic Bayes simply never appears. Sit with that: the entire reason grid/conjugate methods choke on the evidence is sidestepped because MCMC only ever looks at *ratios* of densities at two points, and the constant divides out.

> 💡 **Intuition:** The acceptance rule says: *always* move to a more probable candidate ($R\ge1$, accept with prob 1); move to a *less* probable candidate only sometimes, with probability equal to how much less probable it is. That "sometimes" is what stops the chain from collapsing onto the mode — it lets the walker spend the right amount of time in the lower-density regions, which is exactly what's needed to represent the *whole* distribution, not just its peak.

### Symmetric proposals: Metropolis (1953) vs Hastings (1970)

If the proposal is **symmetric** — $q(\theta'\mid\theta) = q(\theta\mid\theta')$, true for a Gaussian random walk where $\theta' = \theta + \mathcal{N}(0,\Sigma_{\text{prop}})$ — the $q$ terms cancel too, leaving the original 1953 **Metropolis** rule:

$$
\alpha = \min\!\left(1,\ \frac{\tilde p(\theta')}{\tilde p(\theta)}\right).
$$

Hastings' 1970 generalization keeps the $q$-ratio so you can use *asymmetric* proposals (e.g., proposals that respect a boundary). For everything in this chapter we use a symmetric Gaussian random walk, so we're doing plain Metropolis.

> 📜 **Citation/Origin:** Metropolis, Rosenbluth, Rosenbluth, Teller & Teller (1953) invented this at Los Alamos to simulate equilibrium states of interacting particles — *physics*, not statistics. W. K. Hastings (1970) generalized it to arbitrary proposals and brought it into statistics. The phrase "Monte Carlo" itself is a Los Alamos code name, after the casino. We are, quite literally, still running a 1953 physics algorithm — just with much better proposals (HMC, also from physics).

### Hand-rolled Metropolis on a 2-D correlated Gaussian — full code, nothing hidden

Let me make this real. Our target is a **2-D correlated Gaussian**: a mean-zero bivariate normal with strong correlation $\rho=0.9$. We know its true mean and covariance, so we can check the sampler honestly. Critically, we'll feed the sampler **only the unnormalized log-density** — to prove we never need the constant.

```python
# ---- Target: 2-D correlated Gaussian (we pretend we can't normalize it) ----
rho = 0.9
Sigma = np.array([[1.0, rho],
                  [rho, 1.0]])
Sigma_inv = np.linalg.inv(Sigma)

def log_target(theta):
    """UNNORMALIZED log posterior. Note: no normalizing constant included.
    Only the quadratic form matters; the -0.5*log|2πΣ| term is dropped on purpose
    to prove Metropolis never needs it."""
    return -0.5 * theta @ Sigma_inv @ theta

# ---- Hand-rolled random-walk Metropolis ----
def metropolis(log_target, init, n_samples, prop_sd, rng):
    dim = len(init)
    samples = np.empty((n_samples, dim))
    theta = np.asarray(init, dtype=float)
    log_p = log_target(theta)
    n_accept = 0
    for t in range(n_samples):
        # 1) PROPOSE: symmetric Gaussian random walk
        theta_prop = theta + rng.normal(0.0, prop_sd, size=dim)
        log_p_prop = log_target(theta_prop)
        # 2) ACCEPT/REJECT on the log scale (numerically safe):
        #    accept if log(u) < log_p_prop - log_p  <=>  u < p_prop/p
        log_alpha = log_p_prop - log_p
        if np.log(rng.uniform()) < log_alpha:
            theta, log_p = theta_prop, log_p_prop   # move
            n_accept += 1
        samples[t] = theta                          # record (repeats if rejected)
    return samples, n_accept / n_samples

samples, acc_rate = metropolis(log_target, init=[3.0, -3.0],
                               n_samples=20_000, prop_sd=0.4, rng=rng)
print(f"acceptance rate: {acc_rate:.3f}")
```

A few engineering details worth internalizing because you will re-derive them every time you debug a sampler:

- **We work in log space.** We compare `log(u) < log_p_prop - log_p` instead of `u < p_prop/p`. Densities underflow to 0.0 in floating point with astonishing ease (a log-density of $-800$ is a perfectly ordinary value that exponentiates to exactly zero). Always do MCMC arithmetic on the log scale.
- **Rejection means repetition.** When we reject, we record the *current* state again. The chain genuinely sits still for a step. Those repeats are not a bug — they're how lower-density regions get their correct (smaller) share of visits.
- **`prop_sd` is the only tuning knob** of this sampler, and it controls *everything*. We'll see in a moment that it's a Goldilocks parameter.

> 🩺 **Diagnostic — reading a Metropolis trace.** Plot each coordinate against iteration (a **trace plot**) and plot the 2-D path. Three things tell you the sampler is healthy: (1) the trace looks like a "fuzzy caterpillar" — stationary, no trend, rapidly varying; (2) the 2-D cloud fills the true elliptical contours; (3) the **acceptance rate** is moderate (for random-walk Metropolis the theoretically optimal rate is ≈ **0.234** in high dimensions, ≈ 0.44 in 1-D — surprisingly low, because high acceptance means timid steps).

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
burn = 2000                              # discard warmup transient
s = samples[burn:]

axes[0].plot(samples[:, 0], lw=0.5)      # trace of theta_1
axes[0].axvline(burn, color="k", ls="--")
axes[0].set_title(r"trace of $\theta_1$ (caterpillar after warmup)")

axes[1].plot(s[:, 0], s[:, 1], lw=0.3, alpha=0.5)
axes[1].scatter(s[::50, 0], s[::50, 1], s=6, alpha=0.4)
axes[1].set_title("random walk through the correlated ellipse")

az.plot_autocorr(s[:, 0], ax=axes[2])    # ArviZ on a raw numpy array
axes[2].set_title(r"autocorrelation of $\theta_1$")
plt.tight_layout()
```

The trace should look stationary after warmup. The 2-D path will visibly *crawl* along the diagonal of the correlated ellipse — small step after small step, like an ant walking a tightrope. The autocorrelation plot shows the price of that crawl: correlations that decay slowly, meaning many consecutive draws carry nearly the same information. Hold onto that image of the slow diagonal crawl; it's the seed of the failure we're about to dramatize.

### The Goldilocks step size

The proposal scale `prop_sd` has to be *just right*:

| `prop_sd` | Behavior | Acceptance | Result |
|---|---|---|---|
| **Too small** (0.02) | Tiny tiptoe steps | Very high (≈0.98) | Almost every move accepted, but each move is microscopic → glacial exploration, huge autocorrelation |
| **Just right** (≈0.4–1.0 here) | Steps comparable to the posterior's spread | Moderate (≈0.3–0.5) | Good mixing |
| **Too large** (8.0) | Giant leaps | Very low (≈0.02) | Almost every proposal lands in the low-density middle-of-nowhere and is rejected → chain freezes in place |

Both failure modes give terrible **effective sample size**. And here's the cruel part, foreshadowed: *the window of "just right" shrinks as dimension grows.* That shrinkage is the whole story of §3.

---

## 3. Gibbs sampling: free acceptance, but it zig-zags

Before we diagnose the failure of random-walk methods, meet the other classical workhorse, because it has the *same* disease in a different guise.

**Gibbs sampling** is a special case of Metropolis–Hastings where the proposal *is* the exact **full conditional** distribution — the distribution of one parameter given all the others and the data:

$$
\theta_i^{(t+1)} \sim p\big(\theta_i \,\big|\, \theta_{-i}^{(t)},\, y\big),
$$

where $\theta_{-i}$ means "all parameters except $\theta_i$." You sweep through the parameters one at a time, each time drawing the chosen one fresh from its full conditional while holding the rest fixed. The remarkable fact:

> 🧮 **The math — Gibbs acceptance is always 1.** If you propose $\theta_i'$ from the exact conditional $q = p(\theta_i\mid\theta_{-i},y)$ and use the Metropolis–Hastings ratio, every term cancels and $\alpha=1$. Each update is a guaranteed move. Intuitively: you're sampling exactly from the right conditional slice, so there's nothing to correct. Gibbs is "MH with a perfect, never-rejected proposal — *along one axis at a time*."

The cost: you need those full conditionals in closed form, which essentially requires **conjugacy** (Chapter 02). When you have it, Gibbs is elegant. Here is the canonical conjugate example — a Normal with unknown mean and variance under conjugate priors — where both conditionals are standard distributions:

```python
# Conjugate Gibbs for Normal data with unknown mu and tau (precision = 1/variance).
# Priors:  mu ~ Normal(mu0, 1/(kappa0*tau)) ,  tau ~ Gamma(a0, b0)  (semi-conjugate)
# Full conditionals (textbook conjugate updates):
#   mu  | tau, y ~ Normal( weighted mean , precision = kappa0*tau + n*tau )
#   tau | mu , y ~ Gamma( a0 + n/2 , b0 + 0.5*sum((y-mu)^2) + 0.5*kappa0*(mu-mu0)^2 )
y = rng.normal(loc=5.0, scale=2.0, size=40)     # synthetic data, truth: mu=5, sd=2
n = len(y); ybar = y.mean()
mu0, kappa0, a0, b0 = 0.0, 1.0, 1.0, 1.0

def gibbs_normal(n_iter, rng):
    mu, tau = 0.0, 1.0
    out = np.empty((n_iter, 2))
    for t in range(n_iter):
        # --- draw mu | tau, y ---
        prec = kappa0 * tau + n * tau
        mean = (kappa0 * tau * mu0 + n * tau * ybar) / prec
        mu = rng.normal(mean, 1.0 / np.sqrt(prec))
        # --- draw tau | mu, y ---
        a_post = a0 + n / 2
        b_post = b0 + 0.5 * np.sum((y - mu) ** 2) + 0.5 * kappa0 * (mu - mu0) ** 2
        tau = rng.gamma(a_post, 1.0 / b_post)       # numpy: shape, scale=1/rate
        out[t] = (mu, 1.0 / np.sqrt(tau))           # store mu and sd
    return out

gibbs_draws = gibbs_normal(5000, rng)
print("posterior mean of mu:", gibbs_draws[2000:, 0].mean().round(3))
print("posterior mean of sd:", gibbs_draws[2000:, 1].mean().round(3))
```

This recovers $\mu\approx5$, $\sigma\approx2$ with zero rejected moves. Beautiful — when it applies.

### Why Gibbs zig-zags (and why this matters)

Gibbs moves **one axis at a time**: it updates $\theta_1$ along the vertical, then $\theta_2$ along the horizontal, then $\theta_1$ again, and so on. When the parameters are nearly *independent* in the posterior (axis-aligned ellipse), this is great — each conditional move can be large. But when they are **strongly correlated** (a tilted, cigar-shaped ellipse), each conditional is a *narrow slice*, so each move is tiny, and the chain inches up the diagonal in a maddening **staircase**:

```
   strong posterior correlation  →  Gibbs staircases up the diagonal
        θ2 ↑
           │        ┌─┐
           │      ┌─┘ ╲  ← each step is short because, holding θ1 fixed,
           │    ┌─┘    ╲    the conditional for θ2 is a thin slice
           │  ┌─┘       ╲
           │┌─┘ (and vice-versa)
           └──────────────────→ θ1
```

The result is exactly the slow diagonal crawl we saw with random-walk Metropolis — just produced by a different mechanism. **Correlation in the posterior is the enemy of axis-by-axis samplers.** This is precisely why the BUGS/JAGS generation of Bayesian software (built on Gibbs) struggled with the hierarchical and correlated models we most want to fit, and a big part of *why HMC won*: HMC moves along the *natural* direction of the posterior, not along the coordinate axes.

> 🔧 **In practice:** PyMC v5 has **no `pm.Gibbs` class**. Gibbs survives in PyMC only as the discrete steppers `pm.BinaryGibbsMetropolis` and `pm.CategoricalGibbsMetropolis`, which PyMC auto-assigns to discrete parameters inside a `CompoundStep` while NUTS handles the continuous ones. For continuous models you will essentially never hand-roll Gibbs — you'll use NUTS. We teach Gibbs as a *concept* (and a window into why correlation hurts), not as a tool you'll reach for. (See the PyMC "Compound Steps in Sampling" example.)

---

## 4. The typical set: why random-walk MCMC dies in high dimensions

This is the conceptual heart of the chapter, and the single idea that converts "MCMC is some sampling thing" into "I understand why we need gradients." I'm going to follow Betancourt's framing carefully, because it's the clearest, and then I'm going to *prove the consequence to you numerically* — watch the effective sample size of our hand-rolled Metropolis sampler collapse as we crank up the dimension.

### Density is not mass: the war between density and volume

Here is the seductive wrong intuition that everyone starts with: *"The posterior is biggest at the mode, so the samples should pile up near the mode."* In high dimensions this is **false**, and the reason is one of the most beautiful facts in probability.

The expected value of any function is a product of two things integrated together — the **density** $p(\theta)$ and the **volume** $d\theta$:

$$
\mathbb{E}_p[f] = \int f(\theta)\,\underbrace{p(\theta)}_{\text{density}}\,\underbrace{d\theta}_{\text{volume}}.
$$

The contribution of a region to any expectation is *density × volume*. Now think about a standard Gaussian in $d$ dimensions and ask: how much *volume* is there at radius $r$ from the mode? The volume of a thin spherical shell at radius $r$ grows like its surface area, $\propto r^{d-1}$. In $d=200$ dimensions, the volume at radius $r=1.1$ versus $r=1.0$ differs by a factor of $1.1^{199}\approx 1.7\times10^{8}$ — there is *vastly* more room a little farther out.

Meanwhile the **density** of a standard Gaussian falls off like $e^{-r^2/2}$ — it's highest at the mode and decays as you move out. So we have a war:

```
   density  p(r) ∝ exp(-r²/2)          ↓ falls   as r grows
   volume   V(r) ∝ r^{d-1}             ↑ explodes as r grows  (in high d)
   ─────────────────────────────────────────────────────────────
   MASS  ∝  density × volume   →   peaks at a NONZERO radius,
                                    on a thin shell, NOT at the mode
```

The **product** — the actual probability *mass* — peaks where these two opposing trends balance, at a radius $r \approx \sqrt{d}$ for a standard $d$-dimensional Gaussian. That shell of near-constant radius where essentially all the probability mass concentrates is the **typical set**.

> 🧮 **The math — the radius where mass lives.** For $\theta\sim\mathcal N(0, I_d)$, the squared distance from the mode is $\lVert\theta\rVert^2 = \sum_{i=1}^d \theta_i^2 \sim \chi^2_d$, which has mean $d$ and standard deviation $\sqrt{2d}$. So $\lVert\theta\rVert \approx \sqrt{d}$ with a *relative* spread $\sim 1/\sqrt{2d}$ that shrinks as $d$ grows. The samples live at radius $\sqrt d$, and the shell gets *thinner* (relative to its radius) as dimension grows. **The mode, at radius 0, contains a negligible fraction of the mass** — in $d=200$ the typical radius is ≈ 14.1 and the shell is razor-thin around it. If you stand at the mode you are in a region of almost zero probability *mass*, even though it's the point of highest density.

> 💡 **Intuition (concentration of measure):** This is "concentration of measure," and it's deeply counterintuitive coming from 1-D and 2-D where the mode and the mass coincide. The honest mental image: in high dimensions a Gaussian is not a fuzzy blob centered at the origin — it's a **thin soap-bubble shell** at radius $\sqrt d$, and almost all the probability is *on the bubble*, none in the middle. Your samples should live on the bubble.

### Why random-walk proposals can't cope

Now reconsider random-walk Metropolis on the bubble. The proposal is **isotropic** — a symmetric Gaussian blob centered at the current point — utterly blind to the shell's geometry. Consider its options:

- **Step toward the mode (inward).** Higher density there, but you're leaving the high-volume shell for a region with almost no mass. The chain is pulled by density toward a place it should barely visit.
- **Step outward.** More volume, but the density has dropped, so the candidate's *unnormalized posterior* is lower → likely rejected.
- **Step along the shell (tangentially).** This is what you *want*, but an isotropic proposal has no idea which direction is "along the shell." Only a vanishing fraction of random directions happen to stay on the thin shell.

So the chain faces a brutal trade-off that gets worse with $d$:

- Make the step **small enough** to usually land back on the thin shell → moves are tiny → you diffuse along the shell at a crawl. Because the typical set has a circumference that grows with $d$ and your steps stay small, you need $O(d)$ or more steps just to traverse it, and consecutive samples are crushingly correlated. **ESS collapses.**
- Make the step **large** to cover ground → you overshoot the thin shell into the empty interior or exterior → rejected almost always → the chain freezes.

The "Goldilocks" step that we met in 2-D *shrinks toward zero* as $d$ grows, and even at its best the efficiency degrades. This is not a tuning failure you can fix by being clever with `prop_sd`. It is the **curse of dimensionality** for random-walk MCMC, and it's why Gibbs (which has the analogous problem under correlation) and random-walk Metropolis are essentially unusable for the high-dimensional, correlated posteriors that real Bayesian models produce.

> 📜 **Citation/Origin:** This whole picture — typical set, concentration of measure, why random-walk MCMC fails, why you must *follow the geometry* — is from Michael Betancourt's "A Conceptual Introduction to Hamiltonian Monte Carlo" (arXiv:1701.02434), §2–3, and his "Markov Chain Monte Carlo in Practice" case study. If you read one thing alongside this chapter, read §2 of 1701.02434. It's the clearest exposition in the field.

### Demonstrate it: watch Metropolis ESS collapse with dimension

Talk is cheap. Let's *prove* the dimension scaling. We'll target an isotropic standard Gaussian in $d = 2, 10, 50, 200$ dimensions — the easiest possible target, no correlation at all — and run our hand-rolled Metropolis sampler with a *fixed budget* of draws. We'll tune `prop_sd` toward the optimal acceptance rate at each dimension and measure the **effective sample size** (ESS) of the first coordinate via ArviZ. ESS is the headline number: it estimates how many *truly independent* draws your correlated chain is worth.

```python
def gaussian_logpdf_iso(theta):
    return -0.5 * np.dot(theta, theta)   # unnormalized standard Gaussian, any dim

def ess_at_dimension(d, n_samples=20_000, rng=None):
    # Rule of thumb: random-walk Metropolis is ~efficient when the step scales
    # like 2.38/sqrt(d) for an isotropic target (Roberts-Gelman-Gilks scaling).
    prop_sd = 2.38 / np.sqrt(d)
    init = np.zeros(d)
    samples, acc = metropolis(gaussian_logpdf_iso, init, n_samples, prop_sd, rng)
    burn = n_samples // 4
    chain = samples[burn:, 0]                       # track coordinate 0
    # az.ess expects shape (chain, draw); wrap single chain:
    ess = float(az.ess(chain[np.newaxis, :]))
    return prop_sd, acc, ess

print(f"{'dim':>5} {'prop_sd':>9} {'accept':>8} {'ESS(coord0)':>12} "
      f"{'ESS/iter':>9}")
for d in [2, 10, 50, 200]:
    prop_sd, acc, ess = ess_at_dimension(d, rng=np.random.default_rng(d))
    print(f"{d:>5} {prop_sd:>9.3f} {acc:>8.3f} {ess:>12.1f} {ess/15000:>9.4f}")
```

You will see output with the unmistakable shape (your exact numbers will vary with the seed, but the *trend* is the lesson):

```
  dim   prop_sd   accept  ESS(coord0)  ESS/iter
    2     1.683    0.547       ~2200    ~0.147
   10     0.753    0.357        ~520    ~0.035
   50     0.337    0.282        ~120    ~0.008
  200     0.168    0.255         ~28    ~0.002
```

Read that table slowly, because it *is* the curse of dimensionality made of numbers. With the same 15,000 post-warmup draws:

- In $d=2$ we extract roughly **2,200 effective samples** — the chain is worth ~15% of its length.
- By $d=200$ we extract roughly **28** — the chain is worth ~0.2% of its length, a **~80× collapse** in efficiency, on the *easiest possible target* (no correlation, perfectly scaled steps). To get 400 effective samples in $d=200$ you'd need to run on the order of $10^5$–$10^6$ draws, and on a *correlated* target it's far worse.

That collapse is *intrinsic to random-walk exploration*. No amount of step-size tuning fixes it — we already used the optimal $2.38/\sqrt d$ scaling. The only escape is to stop guessing directions and instead **use information about the shape of the posterior to move along the typical set on purpose.** That information is the **gradient** of the log-density, $\nabla\log p(\theta\mid y)$ — it points "uphill" in density, and combined with momentum it lets a trajectory glide *tangentially* around the shell instead of diffusing across it. That is exactly what Hamiltonian Monte Carlo does, and it's where we go next.

> 🩺 **Diagnostic — the number that matters is ESS, not draws.** Burn this in now because it governs all of Chapter 07: a chain's *length* is nearly meaningless; its **effective sample size** is what determines the Monte Carlo error of your estimates. The course threshold (set in Ch 07) is **ESS_bulk and ESS_tail ≳ 400** (≳ 100 per chain) for stable reporting. A 100,000-draw random-walk chain in 200-D with ESS ≈ 28 is *worse* than a 1,000-draw NUTS chain with ESS ≈ 900. We will always report ESS, never raw draws, as the measure of how much information we actually have.

---

## 5. Hamiltonian Monte Carlo: borrow momentum, ride the gradient

We've established the problem (the typical set is a thin shell; isotropic random walks can't follow it) and the cure (use the gradient to move *along* it). HMC is the most elegant way to turn that cure into an algorithm, and — like Metropolis — it comes straight from physics. The physical metaphor is not a loose analogy; it's the literal mathematics.

> 💡 **Intuition — the hockey puck on a frictionless rink.** Picture the negative log-posterior $-\log p(\theta\mid y)$ as a landscape of *potential energy*: valleys are high-posterior regions, hills are low-posterior regions. Now place a frictionless hockey puck somewhere on this landscape and **flick it in a random direction with a random speed**. It will glide — speeding up as it falls into valleys, slowing as it climbs — tracing a long, sweeping path that naturally hugs the contours of the landscape. Because energy is conserved, the puck doesn't just fall to the bottom (the mode) and stop; it orbits, spending time across the whole valley. That gliding orbit is exactly a path *along the typical set*. HMC flicks the puck, lets it glide for a while, records where it stops, re-flicks, and repeats. Every flick + glide is one HMC proposal, and a single glide can carry you clear across the posterior — no diffusion, no crawl.

### Step 1: augment with momentum

The genius move is to **add auxiliary variables**. For our position variable $\theta$ (the parameters we care about), introduce a **momentum** variable $r$ of the same dimension, drawn from a Gaussian:

$$
r \sim \mathcal N(0, M),
$$

where $M$ is a positive-definite **mass matrix** (the "metric," tuned later; think $M=I$ for now). We now work in the *joint* space $(\theta, r)$ with the joint density

$$
p(\theta, r) \propto p(\theta\mid y)\,\mathcal N(r\mid 0, M) = \exp\!\big(\log p(\theta\mid y) - \tfrac12 r^\top M^{-1} r\big).
$$

Why on earth would *adding* variables help? Because of a property that's almost too convenient: $\theta$ and $r$ are **independent** in the joint (the density factorizes). So if we generate samples $(\theta_n, r_n)$ from the joint and then simply **discard the $r$'s**, the marginal of the $\theta$'s is *exactly* our posterior. We haven't changed the target at all — we've just lifted it into a bigger space where a better dynamics is available.

### Step 2: the Hamiltonian

Define, by analogy to physics, the **potential energy** $U$ (from position) and **kinetic energy** $K$ (from momentum):

$$
U(\theta) = -\log p(\theta\mid y) \quad(\text{up to a constant}), \qquad K(r) = \tfrac12\, r^\top M^{-1} r.
$$

Their sum is the **Hamiltonian** (the total energy):

$$
H(\theta, r) = U(\theta) + K(r).
$$

Notice the joint density is precisely $p(\theta, r) \propto e^{-H(\theta, r)}$. High-probability regions are *low-energy* regions. Sampling the joint $\Leftrightarrow$ exploring level sets of $H$.

### Step 3: Hamilton's equations — how the puck moves

Physics tells us how a system at energy $H$ evolves in time. The trajectory $(\theta(t), r(t))$ obeys **Hamilton's equations**:

$$
\frac{d\theta}{dt} = \frac{\partial H}{\partial r} = M^{-1} r, \qquad
\frac{dr}{dt} = -\frac{\partial H}{\partial \theta} = -\nabla U(\theta) = \nabla \log p(\theta\mid y).
$$

```
   dθ/dt =  M⁻¹ r            ← position drifts in the direction of momentum
   dr/dt =  ∇ log p(θ|y)     ← momentum is kicked by the gradient of the log-posterior
```

**There it is — the gradient.** The momentum is continuously pushed by $\nabla\log p$, the very quantity we said points "along the shell." This is the mathematical reason HMC defeats the curse of dimensionality: the dynamics *use the shape of the posterior* to steer, rather than proposing blindly. The puck glides along the typical set because the gradient is steering it there.

Exact Hamiltonian dynamics have two miraculous properties that make them perfect for MCMC:

1. **Energy is conserved:** $H(\theta(t), r(t))$ is constant along the trajectory. So $e^{-H}$ — the joint density — is constant too. If we could integrate exactly, every proposed point would have *identical* joint density to the start, and the Metropolis acceptance ratio would be exactly 1. **Every move accepted, and the move can be huge.**
2. **Volume is preserved** (Liouville's theorem): the flow neither compresses nor expands phase-space volume. This is what lets us use the dynamics as a valid MCMC proposal without messy Jacobian corrections.

> 🧮 **The math — why conservation gives acceptance 1.** Along an exact trajectory, $\frac{dH}{dt} = \frac{\partial H}{\partial\theta}\dot\theta + \frac{\partial H}{\partial r}\dot r = \frac{\partial H}{\partial\theta}\frac{\partial H}{\partial r} + \frac{\partial H}{\partial r}\big(-\frac{\partial H}{\partial\theta}\big) = 0$. The two terms cancel by construction. Energy is conserved, so $H_{\text{end}} = H_{\text{start}}$, so $e^{-H_{\text{end}}}/e^{-H_{\text{start}}} = 1$. With exact dynamics HMC is a perfect sampler. The only reason it isn't perfect in practice is that we cannot solve those differential equations exactly on a computer — which brings us to leapfrog.

### Step 4: the leapfrog integrator — and why *not* Euler

We must discretize Hamilton's equations to simulate them. The obvious choice, the **Euler method**, is a disaster here: it does *not* preserve volume and it lets the energy drift systematically, so the simulated trajectory spirals outward off the energy level set and ruins acceptance. We need an integrator that respects the geometric structure — one that is **symplectic** (volume-preserving) and **time-reversible**. The standard choice is the **leapfrog** (Störmer–Verlet) integrator. One leapfrog step of size $\epsilon$ is a **kick–drift–kick**:

$$
r_{t+\epsilon/2} = r_t + \tfrac{\epsilon}{2}\,\nabla\log p(\theta_t) \qquad\text{(half-step momentum "kick")}
$$
$$
\theta_{t+\epsilon} = \theta_t + \epsilon\, M^{-1} r_{t+\epsilon/2} \qquad\text{(full-step position "drift")}
$$
$$
r_{t+\epsilon} = r_{t+\epsilon/2} + \tfrac{\epsilon}{2}\,\nabla\log p(\theta_{t+\epsilon}) \qquad\text{(half-step momentum "kick")}
$$

```
   kick (½ε)  →  drift (ε)  →  kick (½ε)
   update r    update θ      update r
   using ∇    using new     using NEW ∇
   at θ_t      momentum      at θ_{t+ε}
```

You take $L$ such steps to build a trajectory of length $L\epsilon$. The half-step structure is the secret: by splitting the momentum update into two halves around a full position update, the map becomes **symplectic** — it preserves phase-space volume *exactly* (not approximately), and it is **time-reversible** (run it backward and you retrace your path). Those two properties are exactly what we need for a valid, correction-friendly MCMC proposal. The price is a controlled error: leapfrog conserves energy only approximately, with **local (per-step) error $O(\epsilon^3)$ and global error $O(\epsilon^2)$** over a fixed integration time (it is a *second-order* integrator) that, crucially, *oscillates around the true energy rather than drifting away from it* — so even long trajectories stay near the right energy level set. That bounded, oscillating error is the property Euler lacks.

> 🧮 **The math — why second-order and why "leapfrog."** The name comes from position and momentum updates "leapfrogging" over each other on a half-step-offset grid. Because the step is a symmetric composition (half-kick ∘ drift ∘ half-kick), all the even-order error terms cancel in the local expansion, leaving leading **local (per-step) error $O(\epsilon^3)$**; accumulated over the $\sim 1/\epsilon$ steps of a fixed-length trajectory this gives **global error $O(\epsilon^2)$** — i.e. leapfrog is a *second-order* integrator, far better than Euler's $O(\epsilon)$, with the volume preservation thrown in for free. See Neal (2011), §5.2, for the clean derivation.

### Step 5: the Metropolis correction makes it *exact*

Because leapfrog only approximately conserves energy, the proposed end-state $(\theta^*, r^*)$ has a *slightly* different Hamiltonian than the start, $H^* \neq H_0$. If we just accepted it we'd be sampling from a subtly wrong distribution (biased by the integrator). The fix is a **Metropolis accept/reject** on the *energy error*:

$$
\alpha = \min\!\Big(1,\ \exp\big(H(\theta_0, r_0) - H(\theta^*, r^*)\big)\Big) = \min\!\big(1,\ e^{-\Delta H}\big).
$$

Because leapfrog is symplectic and (with a momentum flip) reversible, this Metropolis step makes the whole thing satisfy the right balance condition, and the **discretization bias is removed exactly** — HMC samples from the *exact* posterior, not an approximation. When the integrator is accurate, $\Delta H \approx 0$ and acceptance is near 1, which is *why HMC can take enormous steps and still accept them.* The accept probability staying high *is* the signal that the integrator is tracking the true dynamics.

> ⚠️ **Pitfall — HMC's correctness argument is *not* plain detailed balance.** This is the subtlety we flagged in §1. A single deterministic leapfrog trajectory is not reversible on its own — momenta point "forward." HMC restores reversibility by *negating the momentum* at the end of the trajectory (which leaves $K$ unchanged since $K(r)=K(-r)$) so that the proposal map is its own inverse. That momentum-flip + volume-preservation is what licenses the Metropolis correction. Don't try to verify HMC with the naïve $p(\theta)T(\theta'|\theta)=p(\theta')T(\theta|\theta')$ check; the right argument lives in the joint $(\theta,r)$ space with the flip. Neal (2011) §5.3 has the full proof.

### Step 6: momentum refresh makes it ergodic

One trajectory at fixed energy explores only a single level set of $H$ — a single "orbit." To explore the *whole* posterior we need to hop between energy levels. That's what the **momentum refresh** does: at the start of every HMC iteration we **throw away the old momentum and draw a fresh $r\sim\mathcal N(0,M)$.** Since $\theta$ and $r$ are independent in the joint, resampling $r$ leaves the $\theta$-marginal invariant — it's a legal Gibbs-style update of the momentum block — but it changes the total energy, kicking the system onto a new level set for the next glide. Refresh → glide → accept/reject → refresh → glide → … Each refresh randomizes the direction of the next sweep, which is what makes the chain irreducible and ergodic.

> 💡 **Intuition — why this beats random walk, in one sentence.** Random-walk Metropolis takes one tiny, blind step per iteration; HMC takes *one coherent glide of hundreds of gradient-informed leapfrog steps* per iteration, traveling a long, low-rejection path *along* the typical set. That's the difference between an 80× efficiency collapse in 200-D and a sampler whose efficiency barely degrades with dimension.

### Full HMC in one box (pseudocode)

```python
# pseudocode — one HMC transition (fixed L, fixed eps, mass matrix M = I)
def hmc_step(theta, log_p, grad_log_p, eps, L, rng):
    r0 = rng.normal(size=theta.shape)          # 1) refresh momentum  r ~ N(0, I)
    r = r0.copy()
    theta_new = theta.copy()
    r += 0.5 * eps * grad_log_p(theta_new)     # 2) leapfrog: first half-kick
    for _ in range(L):
        theta_new += eps * r                   #    drift (full step), M = I -> M^-1 r = r
        if _ != L - 1:
            r += eps * grad_log_p(theta_new)    #    full kick between steps
    r += 0.5 * eps * grad_log_p(theta_new)     #    final half-kick
    r = -r                                     # 3) flip momentum -> reversible proposal
    # 4) Metropolis correction on the energy error  ΔH = (U+K)_new - (U+K)_old
    H0  = -log_p(theta)     + 0.5 * r0 @ r0
    H1  = -log_p(theta_new) + 0.5 * r  @ r
    if np.log(rng.uniform()) < (H0 - H1):      # accept with prob min(1, exp(-ΔH))
        return theta_new, log_p(theta_new)
    return theta, log_p                        # reject: keep old position
```

Run this on our 2-D correlated Gaussian and you'd see trajectories that **sweep across the entire ellipse in a single proposal**, with acceptance near 1 — no diagonal crawl. The two knobs are the step size $\epsilon$ and the number of steps $L$. And *those two knobs are exactly the pain points* that NUTS and adaptation exist to remove. Pick $\epsilon$ too big and leapfrog goes unstable and rejects; too small and you waste gradient evaluations. Pick $L$ too small and you're back to random-walk-like behavior; too big and the trajectory loops back on itself (a "U-turn") and wastes computation returning to where it started. We need to automate both. Enter NUTS and warmup.

---

## 6. NUTS: the No-U-Turn Sampler

Plain HMC has an awkward hyperparameter: the trajectory length $L$ (how many leapfrog steps before you stop and accept). The right $L$ varies by problem and even by region of the same posterior — too short wastes HMC's superpower, too long literally runs you in circles. Tuning $L$ by hand for every model is exactly the kind of fiddly expertise that keeps Bayesian inference from being usable by non-specialists. The **No-U-Turn Sampler (NUTS)** of Hoffman & Gelman (2014) automates it, and it's what runs by default inside `pm.sample()` for continuous models.

> 💡 **Intuition — when should the glide stop?** A glide is productively exploring as long as it keeps getting *farther* from where it started. The moment the trajectory starts curving back toward its origin — beginning to make a "U-turn" — continuing is wasted effort; you'd just retrace ground. So NUTS's rule is: **keep extending the trajectory until it starts to double back on itself, then stop.** No fixed $L$; the trajectory grows exactly as long as it's useful, adapting its length to the local geometry.

### Recursive doubling builds the trajectory

NUTS builds the trajectory by **recursive doubling**. Starting from the current point with a fresh momentum, it repeatedly doubles the trajectory: simulate 1 leapfrog step, then 2 more, then 4, then 8, …, each time choosing *randomly* whether to extend forward or backward in time (to keep the construction reversible). After $j$ doublings the trajectory is a balanced **binary tree** of $2^j$ leapfrog states. This forward-and-backward, doubling construction is what preserves detailed balance: the trajectory is grown symmetrically in time, not just pushed forward.

### The no-U-turn termination criterion

The doubling stops when the trajectory starts to turn around. Concretely, for a sub-trajectory with leftmost (earliest) endpoint $(\theta^-, r^-)$ and rightmost (latest) endpoint $(\theta^+, r^+)$, the **no-U-turn criterion** says: stop when continuing would *reduce* the distance between the endpoints, i.e. when either end's momentum has begun pointing back toward the other end:

$$
(\theta^+ - \theta^-)\cdot r^- < 0 \qquad\text{or}\qquad (\theta^+ - \theta^-)\cdot r^+ < 0.
$$

```
   ((θ⁺ − θ⁻) · r⁻ < 0)  OR  ((θ⁺ − θ⁻) · r⁺ < 0)   →  U-turn detected, STOP
       └──────┬──────┘                 the displacement vector and an
         displacement                  endpoint momentum point in
         from − to +                   "opposite" directions
```

Read the dot product geometrically: $(\theta^+ - \theta^-)$ is the displacement vector from one end of the trajectory to the other; $r^{\pm}$ is the velocity at an end. When the dot product goes negative, that end is now moving *back toward* the other end — the trajectory has started to curl around. NUTS checks this condition not just on the global endpoints but on **every sub-tree** during doubling, so it can detect U-turns within any balanced sub-trajectory and stop early. (The *generalized* U-turn criterion used with a non-identity mass matrix replaces the plain dot product with one weighted by $M^{-1}$; the idea is identical.)

> 📜 **Citation/Origin:** Hoffman & Gelman, "The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo," JMLR 15 (2014). Algorithm 6 in that paper is the canonical pseudocode combining NUTS with dual-averaging step-size adaptation. The exact sign/indexing of the termination condition has a couple of equivalent statements in the literature — the one above is the standard form; verify against §3 of the paper if you ever need the precise generalized version.

### Slice vs multinomial: how the next state is chosen from the tree

Once the tree is built, NUTS samples the *next* state of the chain from among the trajectory's points in a way that preserves the target. The **original 2014 paper** used a **slice sampler** over the trajectory (introducing an auxiliary slice variable $u$ and keeping states with $e^{-H} > u$). **Modern implementations — Stan, and PyMC's NUTS — use multinomial sampling instead**, drawing the next point from the trajectory with probabilities proportional to $e^{-H}$ (favoring higher-density, lower-energy states), which mixes better and is what you're actually running. So when you read the 2014 paper, mentally substitute "multinomial" for "slice" to match today's software.

### `max_treedepth`: the safety cap

Doubling could in principle run forever if the no-U-turn condition never triggers (e.g., a badly scaled posterior where the trajectory keeps gliding without curling). To bound cost, NUTS caps the number of doublings at **`max_treedepth`**, default **10** — i.e., at most $2^{10} = 1024$ leapfrog steps per draw. Hitting the cap is an **efficiency warning, not a correctness error**: your samples are still valid, but the sampler is being forced to stop trajectories before their natural U-turn, usually because the step size is tiny or the mass matrix is poorly scaled (so trajectories need to be very long to traverse the posterior). The right response is almost never to just raise `max_treedepth`; it's to *fix the geometry* (standardize, improve the metric, reparameterize) so trajectories naturally terminate sooner. We'll see this in the knobs table.

> 🩺 **Diagnostic — `tree_depth` lives in `sample_stats`.** After sampling, `idata.sample_stats["tree_depth"]` records the depth used for every draw. If many draws hit `max_treedepth`, PyMC warns "`Maximum tree depth reached`." Check it with `idata.sample_stats["tree_depth"].max()`. The fuller diagnostic loop is Chapter 07; here, just know the cap exists and what hitting it means.

### Divergences: the truth-tellers (introduced here, deep-dive in Chapter 07)

There's one more thing NUTS (and HMC) gives you for free that random-walk methods cannot: an **honest alarm when it's failing.** When the posterior has a region of very high curvature — the classic example is the *neck* of a funnel, where the scale shrinks dramatically — a fixed step size $\epsilon$ that's fine elsewhere becomes far too large there. The leapfrog integrator goes unstable, the simulated energy $H$ blows up, and $|\Delta H| = |H^* - H_0|$ shoots past a fixed energy-error threshold (on the order of 1000 by default in PyMC — `# verify against current pymc.NUTS`). When that happens, PyMC abandons the trajectory and flags a **divergence**.

```python
# How to find divergences in the InferenceData (verified idiom)
divergent = idata.sample_stats["diverging"]    # boolean (chain, draw)
n_div = int(divergent.sum())                   # count
pct   = float(divergent.mean()) * 100          # percentage of draws
print(f"{n_div} divergences ({pct:.1f}% of draws)")
```

> 💡 **Intuition — divergences are a feature, not a bug.** A random-walk sampler that fails to explore a region *fails silently* — it just never goes there, and you'd never know your inference was biased. HMC, by contrast, *tells you* when it can't follow the geometry: a cluster of divergences in one region is the sampler raising its hand and saying "I am not faithfully exploring *here*, so don't trust expectations that depend on this region." That honesty is one of HMC's greatest gifts. The standard fixes — raise `target_accept` (smaller $\epsilon$, more able to navigate the curvature) and **non-centered reparameterization** (reshape the geometry so the curvature disappears) — are the subject of Chapter 07. We preview non-centering at the very end of this chapter with Neal's funnel.

> ⚠️ **Pitfall:** Occasionally a single divergence is a false positive (numerical bad luck). But the course rule, set in Chapter 07, is unambiguous: **the target is 0 divergences**, and any divergence that *clusters* in a region is real and must be fixed, never ignored or silenced. Do not paper over divergences by raising the divergence threshold.

### Energy and BFMI: the *other* free diagnostic (introduced here, deep-dive in Chapter 07)

Divergences flag *local* curvature failures; there's a complementary, *global* diagnostic that comes from the same Hamiltonian machinery — the **energy**. Each draw carries a total energy $H$ (it's in `idata.sample_stats["energy"]`), and the momentum refresh between draws shuffles that energy. Two distributions matter: the **marginal energy distribution** (the spread of $H$ across all draws) and the **energy-transition distribution** (the per-step changes $\Delta E$ that momentum resampling induces). If the transition distribution is much *narrower* than the marginal, each refresh moves the system only a little along the energy axis, so the sampler crawls through the tails — slow mixing that R̂ and ESS may not loudly flag.

The single number that summarizes this is the **BFMI** (Bayesian Fraction of Missing Information): roughly, how efficiently momentum resampling explores the energy distribution. **BFMI < 0.3 flags a problem** (the bible threshold; some use 0.2), and the cure is almost always reparameterization — the same fix as for divergences, because both are symptoms of geometry the metric can't absorb. The plot is **`az.plot_energy(idata)`**, which overlays the marginal and transition energy distributions and prints the BFMI. We only *define* energy/BFMI here so you recognize the term; the full reading of the energy plot is **Chapter 07**.

---

## 7. Warmup and adaptation: what `tune` actually tunes

We've been hand-waving "pick a good $\epsilon$ and mass matrix $M$." During the **warmup / tuning** phase (`tune` iterations, default 1000, discarded by default), PyMC's NUTS *learns* both automatically. Two things are adapted:

### 7.1 Step size $\epsilon$ via dual averaging — and what `target_accept` does

The step size is adapted using **Nesterov dual averaging** (Hoffman & Gelman, Algorithm 6) so that the running-average Metropolis acceptance probability converges to a user target, **`target_accept`** (the symbol $\delta$ in the paper). The logic is a feedback loop: if recent acceptance is *too high*, the steps are too timid — *increase* $\epsilon$; if acceptance is *too low*, leapfrog is being inaccurate — *decrease* $\epsilon$. After warmup, $\epsilon$ is frozen at its adapted value.

So what *exactly* does raising `target_accept` do? Demanding a higher acceptance rate forces dual averaging to settle on a **smaller** $\epsilon$:

```
   raise target_accept  →  smaller ε  →  more accurate, more careful leapfrog steps
                                      →  fewer divergences (can navigate high curvature)
                                      →  BUT longer trajectories / deeper trees
                                      →  slower sampling (more gradient evals per draw)
```

| `target_accept` | Effect | When to use |
|---|---|---|
| **0.8** (library default) | Larger $\epsilon$, faster, fine for easy posteriors | PyMC's out-of-the-box default |
| **0.9** (this course's house default) | Slightly smaller $\epsilon$, a bit safer | Our standard teaching call; good default for real models |
| **0.95** | Smaller $\epsilon$, fewer divergences | First escalation when you see a few divergences |
| **0.99** | Much smaller $\epsilon$, slow | Hard geometries (funnels) — but if you *need* 0.99 you should probably reparameterize instead |

> ⚠️ **Pitfall — the two defaults you must keep straight.** The **PyMC library default is `target_accept = 0.8`** (it's a NUTS step-method kwarg). **This course's house default is `target_accept = 0.9`**, which is what you'll see in every `pm.sample()` call here and in later chapters. That's a deliberate, conservative *course convention*, not the library's behavior. We state it loudly so the chapter doesn't seem to contradict itself or the PyMC docs: if you call `pm.sample(...)` with no `target_accept`, you get 0.8; we *always pass* 0.9. When fighting divergences we escalate to 0.95/0.99. Past ~0.99 you hit steeply diminishing returns at sharply rising cost — reparameterize instead.

> 🩺 **Diagnostic — "acceptance probability does not match the target."** If PyMC warns the *achieved* acceptance is far from `target_accept`, dual averaging didn't converge — usually too few tuning steps. Fix: increase `tune` (try 2000–3000) and/or set `init="jitter+adapt_diag"` explicitly. The achieved per-draw acceptance lives in `idata.sample_stats["acceptance_rate"]`.

### 7.2 The mass matrix (metric) $M$: diagonal vs dense

The second adapted object is the **mass matrix** $M$ (equivalently the **metric**), which appears in the kinetic energy $K(r)=\tfrac12 r^\top M^{-1} r$ and in the position update $\dot\theta = M^{-1}r$. Its job is to **rescale and rotate the space so the posterior looks isotropic** — a nice round ball — to the sampler. If one parameter has posterior SD 0.01 and another has SD 1000, a single $\epsilon$ cannot suit both; $M$ fixes that by absorbing the scales. Ideally,

$$
M^{-1} \approx \mathrm{Cov}(\theta \mid y),
$$

so that in the transformed coordinates the typical set is a sphere and trajectories glide efficiently in every direction. PyMC adapts $M$ from the *covariance of the tuning draws*, in expanding windows, during warmup. Two flavors:

- **`adapt_diag`** (the default path): learn a **diagonal** $M$ — just per-parameter variances (scales). Cheap, $O(d)$, and fixes the "different parameters live on wildly different scales" problem. Cannot capture *correlations* between parameters.
- **`adapt_full`**: learn the **full dense** $M$ — the entire covariance, capturing parameter *correlations* (it can rotate the space, not just rescale axes). Handles correlated posteriors that a diagonal metric can't, at $O(d^2)$ cost per step and needing more tuning to estimate the covariance reliably.

The `init` argument of `pm.sample()` selects the metric-adaptation scheme. The full list (verified against `pymc.init_nuts`): `"auto"`, `"adapt_diag"`, `"jitter+adapt_diag"` (what `"auto"` resolves to), `"adapt_full"`, `"jitter+adapt_full"`, `"advi"`, `"advi+adapt_diag"`, `"advi_map"`, `"map"`, `"jitter+adapt_diag_grad"`.

> 🔧 **In practice — what `jitter+` and the default mean.** The default `init="jitter+adapt_diag"` does three things: (1) start near a reasonable point, (2) add small **uniform jitter in $[-1,1]$** to each chain's starting position so chains begin *dispersed* — this is what makes the between-chain R̂ diagnostic meaningful, (3) adapt a **diagonal** metric. If your model is **slow with low ESS but no divergences**, that's the signature of *correlations a diagonal metric can't capture* — switch to `init="jitter+adapt_full"` (dense metric) or, better, **reparameterize to decorrelate** the posterior. If you pass a hand-built `step=pm.NUTS(...)`, the `init` argument is ignored.

> 🩺 **Diagnostic — give warmup enough room.** $\epsilon$ and $M$ are *both* estimated from the tuning draws. If `tune` is too small, both are mis-set, and you get spurious divergences and low ESS that vanish once you bump `tune` to 2000–3000 on hard models. When a model misbehaves, *increasing `tune`* is one of the first, cheapest things to try — more than raising `draws`, which only adds more (still-correlated) samples without fixing the geometry the sampler learned.

---

## 8. Every `pm.sample()` knob, explained

Here is the verified `pm.sample()` signature (PyMC v5; only `draws` is positional, everything after the `*` is keyword-only). I'll then walk every knob you'll actually touch.

```python
idata = pm.sample(
    draws=1000, *,
    tune=1000,                 # warmup iters (None -> 1000); adapts ε and M; discarded
    chains=4,                  # independent chains (None -> max(cores, 2))
    cores=4,                   # parallel processes (None -> min(n_cpus, 4))
    random_seed=RANDOM_SEED,   # reproducibility; pass an int or list of ints
    init="auto",               # -> "jitter+adapt_diag" (metric init scheme)
    step=None,                 # auto: NUTS for continuous, *Metropolis/Gibbs for discrete
    nuts_sampler="pymc",       # or "numpyro" / "nutpie" / "blackjax" (Ch 17)
    target_accept=0.9,         # NUTS kwarg via **kwargs (course default; library is 0.8)
    max_treedepth=10,          # NUTS kwarg via **kwargs (cap on doublings -> ≤2^10 steps)
    return_inferencedata=True, # returns az.InferenceData (the modern default)
    discard_tuned_samples=True,# drop the warmup draws
    progressbar=True,
)
```

| Knob | What it controls | Course guidance |
|---|---|---|
| **`draws`** | Number of *kept* post-warmup samples per chain. | 1000 is plenty for NUTS (high ESS per draw). Raise only if ESS is short *and* geometry is fine. More draws ≠ better mixing. |
| **`tune`** | Warmup iterations where $\epsilon$ and $M$ are adapted; discarded. | Default 1000. Bump to **2000–3000** for hard models *before* touching anything else. |
| **`chains`** | Independent chains from dispersed starts. | **Always ≥ 4.** Multiple chains are required to compute R̂ and to detect multimodality/non-convergence. (Default = `max(cores, 2)`; we set 4 explicitly.) |
| **`cores`** | OS processes to run chains in parallel. | Set to your physical core count. On Windows/macOS hangs, drop to `cores=1` to debug. |
| **`random_seed`** | Seeds the RNG for reproducibility. | **Always pass `RANDOM_SEED`.** Non-negotiable for reproducible science. |
| **`init`** | Metric-adaptation + start scheme (§7.2). | Leave as `"auto"` (→ `jitter+adapt_diag`) by default; switch to `"jitter+adapt_full"` for stubborn *correlated* posteriors. |
| **`target_accept`** | Dual-averaging acceptance target → sets $\epsilon$ (§7.1). | **House default 0.9** (library 0.8). Escalate 0.95 → 0.99 against divergences; reparameterize past that. |
| **`max_treedepth`** | Cap on NUTS doublings (§6). | Default 10. Raise *only* after fixing the metric/standardization; hitting it is an efficiency warning. |
| **`nuts_sampler`** | Which NUTS backend runs. | `"pymc"` here; `"numpyro"`/`"nutpie"`/`"blackjax"` for speed/GPU — **Chapter 17 (Scaling)**. They accept the same `target_accept`, `tune`, `draws`, `chains`. |
| **`step`** | Manually chosen step method(s). | Leave `None` so PyMC auto-assigns NUTS (+ discrete steppers in a `CompoundStep`). Only set it to mix samplers or hand-tune `pm.NUTS(...)`. |
| **`return_inferencedata`** | Return an `az.InferenceData`. | Keep `True` (the default). **We call the result `idata`, never "trace"** — "trace" was the PyMC v3 name. |
| **`discard_tuned_samples`** | Drop warmup draws. | Keep `True` unless you specifically want to inspect adaptation. |

> 🔧 **In practice — the canonical call.** For essentially every model in this course:
> ```python
> idata = pm.sample(1000, tune=1000, chains=4,
>                   target_accept=0.9, random_seed=RANDOM_SEED)
> ```
> Memorize it. When something breaks, the escalation ladder is: bump `tune` → raise `target_accept` → switch `init` to `adapt_full` → **reparameterize** (the real fix for funnels). That ladder is the spine of Chapter 07.

---

## 9. Worked example: hand-Metropolis vs NUTS on the correlated Gaussian

Let's make the entire argument of this chapter concrete and quantitative on a single named target — our **2-D correlated Gaussian** with $\rho = 0.9$ — by putting our hand-rolled Metropolis sampler head-to-head with PyMC's NUTS and comparing **effective sample size per second**. This is the payoff: the same posterior, two samplers, a measurable efficiency gap. (This is a *synthetic* target with a known truth and therefore no observed data — perfect for isolating the sampler comparison. The full named-dataset workflow with a real likelihood and a posterior predictive check arrives in §11 on Eight Schools.)

### The same target as a PyMC model

```python
import pymc as pm
import arviz as az
import numpy as np

RANDOM_SEED = 8927
rho = 0.9
cov = np.array([[1.0, rho], [rho, 1.0]])

coords = {"dim": ["theta1", "theta2"]}
with pm.Model(coords=coords) as corr_gauss:
    # A 2-D correlated Gaussian as the *posterior* itself (no data; pure target).
    theta = pm.MvNormal("theta", mu=np.zeros(2), cov=cov, dims="dim")
    idata_nuts = pm.sample(
        1000, tune=1000, chains=4,
        target_accept=0.9, random_seed=RANDOM_SEED,
    )
```

### Prior predictive sanity (the workflow habit)

Even on a toy target, build the habit from **Chapter 03 (Choosing, Building, and Checking Priors)**: draw from the model before fitting to confirm it's well-formed and the geometry is what you intend.

```python
with corr_gauss:
    # v5 uses draws=, not samples= (the v3/early-v5 `samples=` alias is removed in current 5.x)
    prior = pm.sample_prior_predictive(draws=500, random_seed=RANDOM_SEED)
# az.plot_pair(prior.prior, var_names=["theta"])  # should show the tilted ρ=0.9 ellipse
```

You should see the familiar tilted ellipse — the strong positive correlation that crippled axis-by-axis Gibbs and slowed random-walk Metropolis to a diagonal crawl.

### Diagnostics on the NUTS fit

```python
print(az.summary(idata_nuts, var_names=["theta"]))
# az.plot_trace(idata_nuts, var_names=["theta"])
# az.plot_pair(idata_nuts, var_names=["theta"], kind="kde")
print("divergences:", int(idata_nuts.sample_stats["diverging"].sum()))
print("max tree_depth:", int(idata_nuts.sample_stats["tree_depth"].max()))
```

`az.summary` returns a table with the posterior `mean`, `sd`, the 94% highest-density interval (`hdi_3%`/`hdi_97%`), `mcse_mean`, `ess_bulk`, `ess_tail`, and `r_hat` per parameter. For this easy target you should see:

- **`r_hat` = 1.00** for both coordinates (chains agree; the threshold is **R̂ ≤ 1.01**).
- **`ess_bulk` and `ess_tail` in the high hundreds to low thousands** *out of 4000 draws* — NUTS extracts most of its draws as effective samples even at $\rho=0.9$, because the adapted dense-ish metric and gradient-guided trajectories slice straight along the correlated ellipse.
- **0 divergences**, `tree_depth` small (≈ 2–4) — short, efficient trajectories.

The `plot_trace` panels should be fuzzy stationary caterpillars; `plot_pair` should fill the tilted ellipse smoothly with no gaps or clumping.

> 🩺 **Diagnostic — reading `az.summary` is muscle memory.** Three columns decide trust: **`r_hat`** (want ≤ 1.01 — did the chains converge to the same place?), **`ess_bulk`** (want ≳ 400 — enough effective samples for the *bulk*, e.g. the mean?), and **`ess_tail`** (want ≳ 400 — enough for *tail* quantiles, e.g. the 97% HDI edge?). `mcse_mean` is the Monte Carlo standard error of the posterior mean — how much of your reported mean is sampling noise. Full diagnostic theory is **Chapter 07**; this is the at-a-glance habit.

### Head-to-head efficiency

Now the comparison that ties the whole chapter together. Run the hand-rolled Metropolis on the *same* correlated target, then compare **ESS** and **ESS per second** against NUTS.

```python
import time

# --- hand Metropolis on the correlated Gaussian (unnormalized log-density) ---
cov_inv = np.linalg.inv(cov)
def log_target_corr(t):
    return -0.5 * t @ cov_inv @ t

rng = np.random.default_rng(RANDOM_SEED)
t0 = time.time()
mh_samples, mh_acc = metropolis(log_target_corr, init=[0.0, 0.0],
                                n_samples=20_000, prop_sd=0.25, rng=rng)
mh_time = time.time() - t0
mh = mh_samples[5000:]                                   # discard warmup
ess_mh = float(az.ess(mh[np.newaxis, :, 0]))             # ESS of theta1

# --- NUTS numbers (already sampled above) ---
ess_nuts = float(az.ess(idata_nuts, var_names=["theta"]).theta.sel(dim="theta1"))

print(f"Metropolis : acc={mh_acc:.2f}  ESS(theta1)={ess_mh:7.1f}  "
      f"time={mh_time:5.2f}s  ESS/s={ess_mh/mh_time:7.1f}")
print(f"NUTS       :            ESS(theta1)={ess_nuts:7.1f}")
```

Indicative result (numbers vary by machine/seed; the *gap* is the point):

```
Metropolis : acc=0.45  ESS(theta1)=  ~310  time= ~0.6s  ESS/s=  ~520
NUTS       :            ESS(theta1)= ~2600
```

Even in this *trivial 2-D* problem, NUTS extracts roughly **8× more effective samples** from far fewer raw draws (4000 vs 20000, a fifth) — and the gap explodes with dimension and correlation, where Metropolis collapses (§4) and NUTS barely flinches. **This is why `pm.sample()` defaults to NUTS, and why we never reach for random-walk Metropolis on a continuous model.** The hand-rolled sampler was a magnificent teaching tool and is the wrong tool for the job.

> 💡 **Intuition — what you just proved.** The 2-D experiment shows the gap *opening*; §4 showed Metropolis *collapsing* with dimension; together they make the case airtight. Random-walk methods are blind to geometry and pay for it with autocorrelation that worsens with dimension and correlation. Gradient-guided NUTS reads the geometry and glides along the typical set. That single difference — *use the gradient* — is the most important practical idea in modern Bayesian computation.

---

## 10. A teaser of where this bites: Neal's funnel

I'll close the technical arc with the model that is *designed* to break HMC, so you've seen the villain before Chapter 07 introduces the cure in full. **Neal's funnel** is a synthetic hierarchical posterior: a global log-scale $v$ controls the scale of nine "local" parameters $x_i$.

$$
v \sim \mathrm{Normal}(0, 3), \qquad x_i \mid v \sim \mathrm{Normal}\!\big(0,\ e^{v/2}\big),\quad i=1,\dots,9.
$$

```python
with pm.Model() as funnel:
    v = pm.Normal("v", mu=0, sigma=3)
    x = pm.Normal("x", mu=0, sigma=pm.math.exp(v / 2), shape=9)
    idata_funnel = pm.sample(
        1000, tune=1000, chains=4,
        target_accept=0.95, random_seed=RANDOM_SEED,  # raised; still won't fully fix it
    )

print("divergences:", int(idata_funnel.sample_stats["diverging"].sum()))
# az.plot_pair(idata_funnel, var_names=["v", "x"], coords={"x_dim_0": [0]},
#              divergences=True)   # newer ArviZ: visuals={"divergence": True}
```

When $v$ is very negative, $e^{v/2}$ is tiny, so the $x_i$ are squeezed into a needle-thin **neck**. The posterior's curvature there is enormous: the scale of $x$ changes by orders of magnitude as $v$ moves. A *single* step size $\epsilon$ cannot be simultaneously small enough to navigate the neck and large enough to be efficient in the wide mouth. Leapfrog goes unstable in the neck, $\Delta H$ explodes, and PyMC reports a **cluster of divergences** — exactly where the geometry is pathological. Plot `v` against `x[0]` with `divergences=True` and you'll see the divergences (highlighted points) **pile up in the funnel's neck**, pointing like an arrow at the region the sampler cannot explore. Notice we already raised `target_accept` to 0.95 and it *still* won't fully fix it — that's the lesson.

> 💡 **Intuition — why this matters far beyond a toy.** The funnel is not academic. **Every centered hierarchical model has a funnel hiding in it** — the relationship between a group-level scale parameter and the group-level effects it controls is exactly this geometry. So this toy is the geometry behind the radon model, the eight-schools model, and essentially every multilevel model you'll build in **Chapter 10 (Hierarchical and Multilevel Models)**. The cure is the **non-centered parameterization** — rewrite the funnel so the sampler sees a clean, decoupled standard-normal geometry instead:
> ```python
> with pm.Model() as funnel_noncentered:
>     v = pm.Normal("v", mu=0, sigma=3)
>     x_raw = pm.Normal("x_raw", mu=0, sigma=1, shape=9)        # standardized
>     x = pm.Deterministic("x", x_raw * pm.math.exp(v / 2))     # rescale after
>     idata = pm.sample(1000, tune=1000, target_accept=0.9,
>                       random_seed=RANDOM_SEED)
> ```
> Same model, *vastly* better geometry — the divergences typically vanish. We introduce **non-centered reparameterization** as a *concept* here; **Chapter 07** explains exactly why it works (it factors the prior so the scale no longer multiplies the curvature the sampler feels), and **Chapter 10** makes it the default for every hierarchical model. Keep the terminology fixed: *non-centered*, the canonical cure for funnels.

---

## 11. End-to-end on a named dataset: Eight Schools with NUTS

Everything so far has been a *synthetic target* with a known truth — perfect for isolating sampler mechanics, but with no real likelihood and so no posterior predictive step. Let's close the loop with the canonical first hierarchical fit, **Eight Schools** (Rubin 1981; the running example in Gelman et al.'s *BDA*), and run the full workflow on it: load data → prior predictive → sample → diagnostics → posterior predictive → interpretation. It's also the real-world face of §10's funnel — Eight Schools *is* a centered hierarchical model, so we get to watch the funnel's divergences appear on actual data and then disappear under the non-centered cure.

The data: eight schools ran a coaching program; each reports an estimated treatment effect $y_j$ with a known standard error $\sigma_j$ (from the size of that school's study). We model a shared underlying effect with a hierarchical Normal.

```python
import numpy as np
import pymc as pm
import arviz as az

RANDOM_SEED = 8927

# --- the data (hardcoded, the canonical Eight Schools numbers) ---
J = 8
y     = np.array([28.,  8., -3.,  7., -1.,  1., 18., 12.])   # estimated effects
sigma = np.array([15., 10., 16., 11.,  9., 11., 10., 18.])   # known standard errors
coords = {"school": [f"school_{j}" for j in range(J)]}
```

### Centered model — the natural-but-treacherous parameterization

The textbook ("centered") way to write it: a population mean $\mu$, a between-school SD $\tau$, and each school's true effect $\theta_j \sim \mathrm{Normal}(\mu, \tau)$, observed through its own measurement noise $\sigma_j$.

```python
with pm.Model(coords=coords) as eight_centered:
    mu    = pm.Normal("mu", mu=0.0, sigma=5.0)
    tau   = pm.HalfCauchy("tau", beta=5.0)
    theta = pm.Normal("theta", mu=mu, sigma=tau, dims="school")   # school effects
    pm.Normal("obs", mu=theta, sigma=sigma, observed=y, dims="school")  # REAL likelihood

    # 1) prior predictive: does the model generate plausible effects BEFORE seeing y?
    #    v5 uses draws=, not samples=
    prior_c = pm.sample_prior_predictive(draws=500, random_seed=RANDOM_SEED)

    # 2) sample (course default target_accept=0.9)
    idata_c = pm.sample(1000, tune=1000, chains=4,
                        target_accept=0.9, random_seed=RANDOM_SEED)

print("centered divergences:",
      int(idata_c.sample_stats["diverging"].sum()))
```

Note the `observed=y`: *this* is the genuine likelihood the synthetic targets in §9–§10 lacked. The prior predictive (`prior_c.prior_predictive["obs"]`) lets you sanity-check that, a priori, the model can produce school effects on roughly the right scale (tens of points), not absurd ones — the Chapter 03 habit, now on real data.

When you sample the centered model you will see PyMC print **a handful of divergences** even at `target_accept=0.9`. That is §10's funnel, live: the geometry between $\tau$ and the $\theta_j$ is exactly the neck-and-mouth funnel, and `az.plot_pair(idata_c, var_names=["tau", "theta"], coords={"school": ["school_0"]}, divergences=True)` shows the divergent draws (highlighted) **clustering at small $\tau$** — the squeezed neck. The fit is biased there: the centered model systematically *fails to explore* the small-$\tau$ region (strong pooling), which is precisely the part of the posterior that matters for whether the schools differ at all.

### Non-centered model — the cure, then the full diagnostic + PPC

Now apply §10's non-centered reparameterization: sample standardized effects $\tilde\theta_j \sim \mathrm{Normal}(0,1)$ and rescale, so the sampler sees a clean isotropic geometry and $\tau$ no longer multiplies the curvature it feels.

```python
with pm.Model(coords=coords) as eight_noncentered:
    mu      = pm.Normal("mu", mu=0.0, sigma=5.0)
    tau     = pm.HalfCauchy("tau", beta=5.0)
    theta_t = pm.Normal("theta_t", mu=0.0, sigma=1.0, dims="school")     # standardized
    theta   = pm.Deterministic("theta", mu + tau * theta_t, dims="school")  # rescale
    pm.Normal("obs", mu=theta, sigma=sigma, observed=y, dims="school")

    idata_nc = pm.sample(1000, tune=1000, chains=4,
                         target_accept=0.9, random_seed=RANDOM_SEED)
    # 3) posterior predictive: simulate new school outcomes from the fitted posterior
    pm.sample_posterior_predictive(idata_nc, extend_inferencedata=True,
                                   random_seed=RANDOM_SEED)

print("non-centered divergences:",
      int(idata_nc.sample_stats["diverging"].sum()))               # expect 0
print(az.summary(idata_nc, var_names=["mu", "tau", "theta"]))
```

What to read, in order — this is the diagnostic muscle memory from §9:

- **Divergences → 0.** The same model, reparameterized, removes the funnel the sampler felt. This is the whole point of §10 made concrete on data: *step-size shrinking (raising `target_accept`) papers over the neck; reparameterization removes it.*
- **`r_hat` ≤ 1.01** for `mu`, `tau`, and every `theta[j]`; **`ess_bulk` and `ess_tail` ≳ 400.** Both should now pass cleanly, where the centered model's `tau` typically shows degraded ESS.
- **Interpretation.** The posterior `mu` sits around 4–5 points with wide uncertainty; `tau`'s posterior has substantial mass near 0, i.e. the data are consistent with *strong pooling* — the eight schools may share nearly one common effect, and the apparently large raw differences ($y$ ranges from −3 to 28) are mostly measurement noise. This shrinkage of the per-school `theta[j]` toward `mu` is the hierarchical model earning its keep — and it is exactly the inference the *centered* model got wrong by failing to explore small $\tau$.

Finally, the **posterior predictive check** — the step the synthetic targets could not have, because they had no likelihood:

```python
# 4) does the fitted model regenerate data that look like the observed y?
az.plot_ppc(idata_nc, var_names=["obs"], num_pp_samples=200)   # observed vs replicated
# or numerically: compare observed y to the predictive spread per school
ppc = idata_nc.posterior_predictive["obs"]                     # (chain, draw, school)
print("observed y      :", y)
print("post-pred mean  :", ppc.mean(dim=("chain", "draw")).values.round(1))
```

`az.plot_ppc` overlays the distribution of replicated `obs` draws against the observed `y`. Because each school's likelihood carries its own fixed $\sigma_j$, the replicated effects should bracket the observed ones with the right per-school spread; gross mismatch (observed points in the tails of the replicates) would signal a misspecified likelihood. Here the observed eight values fall comfortably inside their predictive intervals — the model is adequate. This — *data → prior predictive → sample → diagnostics → posterior predictive → interpretation* — is the complete workflow template every later chapter assumes; Chapter 07 turns the diagnostic steps into a rigorous loop, and Chapter 10 makes non-centered hierarchical models the default.

---

## ⚠️ Common errors & how to fix them

These are the messages `pm.sample()` will actually print, what each *means* mechanistically (given everything above), and the fix. The full diagnostic workflow is **Chapter 07**; this table is your in-the-moment first aid.

| Symptom / message | What it means (mechanism) | Fix (in order) |
|---|---|---|
| `There were N divergences after tuning` | $\epsilon$ too large for a high-curvature region (e.g. a funnel neck); leapfrog goes unstable, $|\Delta H|$ blows past threshold | 1) Raise `target_accept` 0.9 → 0.95 → 0.99 (shrinks $\epsilon$). 2) **Reparameterize non-centered** — the real cure. 3) Tighten/rescale priors; **standardize predictors**. |
| `The acceptance probability does not match the target` | Dual-averaging step-size adaptation didn't converge — usually too few tuning steps | Increase `tune` to 2000–3000; set `init="jitter+adapt_diag"` explicitly. |
| `Maximum tree depth reached` (many `tree_depth == 10`) | Trajectories forced to stop before their natural U-turn — metric poorly scaled or $\epsilon$ tiny, so paths must be very long | Improve the metric (`init="jitter+adapt_full"` if correlated), standardize, raise `tune`; raise `max_treedepth` only *after* those. |
| `The rhat statistic is larger than 1.01` | Chains haven't mixed / multimodality / non-convergence | More `tune` + `draws`; reparameterize; check identifiability (detail in Ch 07). Always run ≥ 4 chains so R̂ exists. |
| `The effective sample size per chain is smaller than 100` | Crushing autocorrelation from poor geometry | Reparameterize, raise `target_accept`, longer/more chains; diagnose with `az.plot_autocorr`. |
| **Slow sampling, low ESS, *zero* divergences** | Strong posterior **correlations** a diagonal metric can't capture (no curvature pathology, just tilt) | `init="jitter+adapt_full"` (dense metric), **or** reparameterize to decorrelate. This is the §4/§9 lesson in the wild. |
| `SamplingError: Initial evaluation ... results in -inf` | Bad start: an impossible value under the prior/likelihood (data outside support, log of a negative, etc.) | Set `initvals=`; fix priors/transforms; check the `observed` data domain (e.g. a `Gamma` likelihood can't see zeros). |
| `BlasError` / multiprocessing hangs (some OS) | Multiprocessing context or BLAS thread contention | `cores=1` to debug; `mp_ctx="spawn"`; `blas_cores=1`. |
| Discrete parameter + NUTS: `Cannot sample` / gradient error | NUTS needs gradients; discrete variables have none | Marginalize the discrete variable (**Ch 13**), or let PyMC auto-assign a `CompoundStep` with `*GibbsMetropolis` for the discrete part. |
| Hand-rolled Metropolis "runs fine" but ESS is tiny in high-D | The typical-set / random-walk-diffusion problem (§4) — *not* a bug | Use NUTS. This is the chapter's whole point: gradients beat blind proposals. |

> ⚠️ **Pitfall — the one-line rule.** When in doubt, the highest-leverage moves are, in order: **standardize your predictors**, **increase `tune`**, **raise `target_accept`**, and **reparameterize (non-centered)**. Three of those four cost you nothing but a keystroke. Reaching for `max_treedepth` or silencing warnings should be near the *bottom* of your list, not the top.

---

## 🧪 Exercises

**1. (Conceptual) The evidence cancels.** Write out the Metropolis acceptance ratio for moving from $\theta$ to $\theta'$ for a posterior $p(\theta\mid y) = \frac{p(y\mid\theta)p(\theta)}{p(y)}$ with a symmetric proposal. Show explicitly that $p(y)$ cancels. In one sentence, explain why this single fact is what makes MCMC possible where conjugate/grid methods choke. *(Hint: §2; the answer connects to the hard integral from Chapter 01.)*

**2. (Coding) Goldilocks step size.** Take the hand-rolled `metropolis` from §2 and run it on the 2-D correlated Gaussian with `prop_sd ∈ {0.02, 0.4, 8.0}`. For each, report the acceptance rate and `az.ess` of $\theta_1$, and plot the trace. Explain which failure mode each extreme exhibits (timid-but-stuck vs leaping-and-rejected) and why the middle value wins. *(Hint: acceptance ≈ 0.98 and acceptance ≈ 0.02 are both bad; the table in §2.)*

**3. (Coding/Numeric) Reproduce the dimension collapse.** Using `ess_at_dimension` from §4, run the experiment for $d \in \{2, 5, 10, 25, 50, 100, 200\}$ at fixed total draws, plot `ESS` (log scale) against $d$, and fit a rough power law. Then *repeat with a fixed `prop_sd = 0.5` for all $d$* (i.e., *don't* use the $2.38/\sqrt d$ scaling) and explain why the collapse is even more dramatic. *(Hint: an un-scaled large step in high-D almost never lands on the thin shell.)*

**4. (Conceptual) Why leapfrog, not Euler.** Explain in your own words why the leapfrog integrator is **symplectic** (volume-preserving) while forward Euler is not, and why that property is essential for HMC's Metropolis correction to yield *exact* posterior samples. What goes wrong, concretely, with a non-symplectic integrator over a long trajectory? *(Hint: §5, Step 4; energy drift vs energy oscillation, and the missing-Jacobian problem.)*

**5. (Coding) NUTS beats your hand sampler — measure it.** Build the §9 comparison yourself: fit the $\rho=0.9$ Gaussian with `pm.sample(... target_accept=0.9)`, fit it with your hand-Metropolis, and report ESS *and* ESS-per-second for both. Then crank the correlation to $\rho = 0.99$ and rerun. Quantify how each sampler degrades. Which one would you trust at $\rho = 0.999$ in 50 dimensions? *(Hint: §9; expect NUTS to barely move while Metropolis falls off a cliff.)*

**6. (Conceptual/Coding) Meet the funnel.** Fit Neal's funnel from §10 at `target_accept=0.8`, then `0.95`, then `0.99`. Count divergences each time with `idata.sample_stats["diverging"].sum()` and locate them with `az.plot_pair(..., var_names=["v","x"], divergences=True)`. Does raising `target_accept` alone ever reach **0** divergences? Then fit the **non-centered** version and recount. Write two sentences on why reparameterizing succeeds where step-size shrinking fails. *(Hint: §10; the curvature is in the *parameterization*, not just the step size. Full treatment: Chapter 07.)*

---

## 📚 Resources & further reading

The two things to read first, in this order:

- **Betancourt, "A Conceptual Introduction to Hamiltonian Monte Carlo"** (arXiv:1701.02434). *The* source for the typical set, concentration of measure, why random-walk MCMC fails, and the geometry of HMC. Read **§2** (computing expectations / typical set) and **§3** (Hamiltonian dynamics). If you read one paper for this chapter, read this. <https://arxiv.org/abs/1701.02434> (PDF: <https://arxiv.org/pdf/1701.02434>)
- **Neal, "MCMC using Hamiltonian dynamics"** (Handbook of MCMC, 2011; arXiv:1206.1901). The definitive, careful treatment of leapfrog, momentum, and the correctness proof. See **§5.2** (leapfrog) and **§5.3** (the HMC algorithm + why the momentum flip gives reversibility). <https://arxiv.org/pdf/1206.1901>

Foundational papers:

- **Hoffman & Gelman, "The No-U-Turn Sampler"** (JMLR 15, 2014). NUTS + dual-averaging step size; **Algorithm 6** is the canonical pseudocode. <https://jmlr.org/papers/v15/hoffman14a.html>
- **Betancourt, "Markov Chain Monte Carlo in Practice"** (case study) — typical set, concentration of measure, and "drifting into the typical set" in plain language. <https://betanalpha.github.io/assets/case_studies/markov_chain_monte_carlo.html> (gentler companion: the "Markov Chain Monte Carlo Basics" case study).

Books (free online where noted):

- **Martin, Kumar & Lao, *Bayesian Modeling and Computation in Python*** — the PyMC/ArviZ companion to this course. Chapters 2 and 11 cover inference engines + HMC/NUTS intuition with runnable PyMC. Free: <https://bayesiancomputationbook.com>
- **McElreath, *Statistical Rethinking* (2nd ed., 2020)** — Chapter 9, "Markov Chain Monte Carlo," builds the same intuition with the "King Markov" island-hopping story and `ulam`/HMC. Free lecture videos accompany it.
- **Gelman et al., *Bayesian Data Analysis* (BDA3)** — Chapters 11–12 on MCMC and HMC, the reference treatment. Free PDF.

Docs (verify every knob against these — they are the source of truth):

- **`pymc.sample` API** — the full signature and defaults (`target_accept` default 0.8, `init="auto"`, `max_treedepth=10`). <https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.sample.html>
- **`pymc.init_nuts`** — exact `init` strings and metric-adaptation semantics (`adapt_diag` vs `adapt_full`, `jitter+`). <https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.init_nuts.html>
- **`pymc.NUTS` step method** — `target_accept`, `max_treedepth`, `step_scale`. <https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.NUTS.html>
- **PyMC example, "Diagnosing Biased Inference with Divergences"** — divergence-access idioms, non-centered eight-schools, `target_accept`/`tune` sweeps. <https://www.pymc.io/projects/examples/en/latest/diagnostics_and_criticism/Diagnosing_biased_Inference_with_Divergences.html>
- **PyMC example, "Sampler Statistics"** — what's inside `sample_stats` (energy, tree_depth, step_size). <https://www.pymc.io/projects/examples/en/latest/diagnostics_and_criticism/sampler-stats.html>
- **PyMC example, "Compound Steps in Sampling"** — how PyMC mixes NUTS with discrete `*GibbsMetropolis` steppers. <https://www.pymc.io/projects/examples/en/latest/samplers/sampling_compound_step.html>
- **Stan Reference Manual, MCMC chapter** — an independent, authoritative description of the metric, step size, and tree depth. <https://mc-stan.org/docs/reference-manual/mcmc.html>
- **ArviZ `plot_energy`** — reading the energy plot and BFMI (preview of Ch 07). <https://python.arviz.org/en/stable/api/generated/arviz.plot_energy.html>

Interactive intuition (highly recommended — *watch* the algorithms move):

- **Chi Feng's MCMC demo** — animate random-walk Metropolis vs HMC vs NUTS on the banana and the funnel, and *see* the diagonal crawl vs the gliding trajectory. <https://chi-feng.github.io/mcmc-demo/>

---

## ➡️ What's next

You now have the *mechanism*: you know what `pm.sample()` is doing, why NUTS uses gradients, what `target_accept` and the mass matrix adjust, and what a divergence is screaming. But running a sampler is not the same as *trusting* it. **Chapter 06 — Inference III: Variational Inference** takes a detour to the other great family of inference engines: instead of sampling the posterior, VI *optimizes* an approximation to it (the ELBO, mean-field and full-rank ADVI, normalizing-flow VI, Pathfinder), trading exactness for speed — and we'll be precise about exactly where VI lies to you and how to catch it. Then **Chapter 07 — MCMC Diagnostics and Debugging** turns every alarm we previewed here into a disciplined workflow: rank-normalized R̂, ESS bulk/tail, MCSE, trace/rank/forest plots, the energy plot and BFMI, and the deep, definitive treatment of divergences, the funnel, and the centered-vs-non-centered reparameterization we just teased. The sampler talks; Chapter 07 teaches you to listen fluently.
