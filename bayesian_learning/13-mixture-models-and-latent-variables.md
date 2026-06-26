# Chapter 13 — Mixture Models & Latent Variables

### When your data is secretly several populations wearing one coat

> *A note from me to you.*
>
> *Up to now we have mostly told the small world a single story: there is one
> data-generating process, with one set of parameters, and every observation is a
> draw from it. That story is a lie we tell on purpose — a useful simplification.
> But sometimes it is the* wrong *lie, and the data will tell you so. You fit a
> tidy Normal and the residuals come out lumpy. You fit a Poisson and there are
> way too many zeros. You plot a histogram and instead of one graceful hump you
> see two camels' backs. Every one of these is the same signal: your data is not
> one population. It is a* mixture *of populations, and you only ever observe the
> blend, never the labels that say which observation came from which.*
>
> *This is the chapter where you learn to model the unseen. We will introduce a
> latent variable — an unobserved group membership — and let the model reason
> about it for us. That single move unlocks an enormous amount: density
> estimation, clustering with honest uncertainty, robust regression, excess-zero
> count models, and (next chapter) infinite mixtures. It also introduces two
> genuinely subtle traps — label switching and the awkward marriage of discrete
> parameters with Hamiltonian Monte Carlo — that have wrecked more student
> notebooks than almost anything else in this course. I want you to leave here
> able to spot them on sight and fix them with one line. Sit with the generative
> story until it feels obvious; everything else follows from it.*

---

## What you'll be able to do after this chapter

- **Think in latent variables**: recognize when an unobserved label or structure
  is the natural way to describe your data, and write the generative story for it.
- **Build a finite Gaussian mixture** in PyMC with `pm.NormalMixture` / `pm.Mixture`,
  put a `Dirichlet` prior on the mixing weights, and fit it on the Old Faithful geyser data.
- **Diagnose and cure label switching** using the `ordered` transform on component
  means — and explain *why* the symmetry exists in the first place.
- **Explain marginalization of discrete latents**: why PyMC's `Mixture` sums out the
  component label automatically so NUTS can run, and reproduce the same thing by hand
  in Stan with `log_sum_exp` / `target +=`.
- **Recover cluster membership post hoc** by computing posterior responsibilities from
  your fitted draws — with calibrated uncertainty, not hard assignments.
- **Model excess zeros** with zero-inflated and hurdle distributions
  (`pm.ZeroInflatedPoisson`, `pm.HurdlePoisson`), and crisply distinguish the two.
- **Choose the number of components** using LOO from Chapter 11 — or know when to give
  up on a fixed count and go nonparametric in Chapter 14.

---

## 1. Latent-variable thinking: modeling the things you cannot see

Let me start with the idea, because the machinery is downstream of it.

A **latent variable** is a quantity in your generative model that is real — it
participates in producing the data — but which you never get to observe directly.
You only see its consequences. The classic example: a measurement that is the sum
of a "signal" and a "noise" you cannot separate. But the latent variable we care
about in this chapter is more structural: it is a **discrete label** that says
*which sub-population an observation belongs to*.

> 💡 **Intuition:** A mixture model is the honest admission that "one population"
> was a convenient fiction. The reading on the thermometer doesn't carry a tag
> saying "I came from the cold morning" or "I came from the warm afternoon." But
> if cold mornings and warm afternoons each have their own typical temperature,
> the *blend* of the two will look like two humps. The latent label is the tag we
> wish we had. We can't see it — so we infer it.

Here is the move that makes all of this work, and it is worth saying slowly. In a
Bayesian model **everything unobserved is a parameter** — there is no hard line
between "parameters" and "latent variables." A regression slope is unobserved; we
put a prior on it and infer its posterior. A cluster label is *also* unobserved;
we can put a prior on it (a categorical distribution over groups) and infer *its*
posterior. The only difference is that the label is discrete and there is one of
them per observation, which makes the bookkeeping and the computation more
interesting. But conceptually it is the same machine you already know:

$$
p(\text{unknowns} \mid \text{data}) \propto p(\text{data} \mid \text{unknowns})\, p(\text{unknowns}).
$$

Once you internalize "the cluster label is just another unknown," a huge swath of
models stops looking like separate tricks and starts looking like one idea seen
from different angles:

- **Clustering / density estimation** — the label says which Gaussian bump you came from.
- **Robust regression** — a label says whether you are a "normal" point or an
  "outlier" point, drawn from a fatter-tailed component. (A Student-$t$ likelihood is
  itself a continuous mixture of Normals — more on that idea in a moment.)
- **Zero-inflated counts** — a label says whether you were a "structural zero"
  (the process that could produce a count was switched off) or an "ordinary count."
- **Missing data** (Chapter 16) — the missing value is a latent variable we infer.
- **Hierarchical models** (Chapter 10) — the group-level effect is a latent variable
  shared by everyone in the group.

> 📜 **Citation/Origin:** McElreath groups mixtures and their zero-inflated cousins
> under the wonderful chapter title *"Monsters and Mixtures"* (*Statistical
> Rethinking* 2e, Ch 12). The framing — that these models are built by gluing
> simpler pieces together with a latent switch — is his, and it is the right
> mental picture to carry through this chapter.

### The finite-mixture generative story

A **finite mixture** with $K$ components has a two-step generative story, and if
you can recite this story you can build any mixture model in this chapter:

1. **Pick a component.** Draw a label $z_n \sim \mathrm{Categorical}(w)$, where
   $w = (w_1, \dots, w_K)$ are the **mixing weights** — non-negative and summing to one
   (a point on the simplex). $w_k$ is the prior probability that any given observation
   came from component $k$.
2. **Draw data from that component.** Given $z_n = k$, draw
   $y_n \sim f(\cdot \mid \theta_k)$, where $f$ is the component family (a Normal, a
   Poisson, whatever) and $\theta_k$ are that component's parameters.

In symbols:

$$
z_n \sim \mathrm{Categorical}(w), \qquad
y_n \mid z_n = k \;\sim\; f(y_n \mid \theta_k).
$$

ASCII, because it is worth seeing the shape of it:

```
        w (weights on the simplex)
          |
          v
    z_n ~ Categorical(w)     <- the latent label (UNOBSERVED)
          |
          v
    y_n ~ f(y | theta_{z_n}) <- the data you actually see
```

Now here is the crucial fact. We never observe $z_n$. So the likelihood of an
observation $y_n$, with the label summed (marginalized) out, is a weighted average
of the component densities:

$$
p(y_n \mid w, \theta) \;=\; \sum_{k=1}^{K} w_k \, f(y_n \mid \theta_k).
$$

> 🧮 **The math:** That sum is the entire definition of a finite mixture *density*.
> Read it as: "the probability of seeing $y_n$ is the probability it came from
> component $k$ (which is $w_k$) times how plausible $y_n$ is under component $k$
> (which is $f(y_n\mid\theta_k)$), added up over all the components it could have
> come from." It is just the law of total probability applied to the latent label.
> Every mixture density — Gaussian, Poisson, whatever — is this same weighted sum.

The picture for $K = 2$ Normals: two bell curves of possibly different heights,
locations, and widths, added together. Where they overlap you get one fused hump;
where they don't you get two distinct peaks. The mixing weights control the
relative heights. A mixture of enough Gaussians can approximate *any sufficiently
regular* density to arbitrary accuracy — under mild conditions Gaussian mixtures are
dense in the space of continuous densities — which is exactly why mixtures are the
workhorse of Bayesian density estimation.

> 💡 **Intuition — the continuous mixture aside.** Mixtures don't have to be finite
> or discrete. A **Student-$t$** distribution is an infinite, continuous mixture of
> Normals: take a Normal whose *variance* is itself random (inverse-gamma
> distributed), integrate the variance out, and you get a $t$. That is why the $t$
> has fat tails — it is hedging across many possible variances. So when you reached
> for `pm.StudentT` for robust regression back in Chapter 9, you were already using
> a mixture. The latent variable there was a per-observation variance you summed
> out. Keep this in your pocket; it makes the "marginalize the latent" idea below
> feel inevitable rather than clever.

---

## 2. Meet the data: Old Faithful, and why one Normal will not do

Let me ground all of this in real data immediately, because mixtures are a model
you should be able to *see*.

Old Faithful is a geyser in Yellowstone. The classic dataset records, for a series
of eruptions, the **duration** of the eruption and the **waiting** time until the
next one. We will focus on the waiting times. Load it from seaborn — this loader is
rock-solid and needs no internet beyond the first cache:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pymc as pm
import arviz as az

RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)

geyser = sns.load_dataset("geyser")     # columns: duration, waiting, kind
x = geyser["waiting"].to_numpy().astype(float)

print(geyser.shape)                     # (272, 3)
print(geyser.head())

fig, ax = plt.subplots()
ax.hist(x, bins=30, density=True, alpha=0.5, color="C0")
ax.set_xlabel("waiting time to next eruption (min)")
ax.set_ylabel("density")
```

> 🩺 **Diagnostic:** When you run this histogram, look at the shape. You will see
> two clearly separated humps — a short-wait cluster around the low fifties and a
> long-wait cluster up near eighty minutes, with a sparse valley between them. That
> valley is the tell. A single Normal *cannot* produce a valley in the middle of
> its own mass; it has exactly one mode. So the data is shouting "I am at least two
> populations." Old Faithful has, roughly, two eruption regimes (short eruptions
> tend to be followed by shorter waits, long ones by longer waits), and the waiting
> times inherit that bimodality.

This is the single best reason to learn mixtures: **bimodal (or multimodal) data
breaks unimodal models silently.** If you fit a lone Normal to `x`, the optimizer
will happily place the mean in the empty valley between the two humps — the single
worst place — and report a variance large enough to "cover" both. Every downstream
prediction will be wrong in the same confident way.

> 💡 **Intuition:** Think of fitting one Normal to two camels as trying to describe
> "a typical camel hump" by standing exactly between two camels and measuring the
> air. You get a number, and the number is meaningless.

We will model `x` as a mixture of $K$ Gaussians and let the data decide where the
humps are and how heavy each is. We start with $K = 2$ because the eyes say two,
then we will let LOO arbitrate between candidate $K$ at the end.

---

## 3. The finite Gaussian mixture in PyMC

### 3.1 The weights: a Dirichlet prior on the simplex

The mixing weights $w = (w_1, \dots, w_K)$ live on the **probability simplex**: each
$w_k \ge 0$ and $\sum_k w_k = 1$. The natural, conjugate-flavored prior for a point
on the simplex is the **Dirichlet distribution**, the multivariate generalization
of the Beta. We write

$$
w \sim \mathrm{Dirichlet}(\alpha_1, \dots, \alpha_K),
$$

and the concentration vector $\alpha$ tunes our prior beliefs about the weights:

- $\alpha = (1, 1, \dots, 1)$ — the **symmetric/uniform** prior. Every weight vector
  on the simplex is equally likely a priori. This is the sensible default and the
  one we will use.
- $\alpha_k$ **all large and equal** (e.g. all $= 10$) — pushes the weights toward
  being *equal* ($1/K$ each); a prior belief that the components are balanced.
- $\alpha_k$ **all small** (e.g. all $= 0.1$, i.e. $< 1$) — pushes mass toward the
  *corners* of the simplex: a belief that a few components dominate and the rest are
  near-empty. This **sparsity-inducing** setting is a poor man's way of asking "use
  fewer components than $K$" and previews the Dirichlet-process idea of Chapter 14.

> 🔧 **In practice:** Start with `np.ones(K)`. It says "I have no idea how the mass
> splits across components," which is honest. Reach for a sparse $\alpha < 1$ only
> when you deliberately want the model to switch components off; reach for large
> equal $\alpha$ only when you have a real reason to expect balance.

> ⚠️ **Pitfall:** The Dirichlet concentration must be strictly positive
> ($\alpha_k > 0$). A zero entry is not a valid Dirichlet and PyMC will error at
> model build / first evaluation. `np.ones(K)` is safe; `np.zeros(K)` is not.

### 3.2 The components and the `NormalMixture` wrapper

For a Gaussian mixture, each component $k$ has a mean $\mu_k$ and a standard
deviation $\sigma_k$ (PyMC, like Stan, parameterizes Normals by **standard
deviation $\sigma$, never variance and never precision** — burn this in; BUGS/JAGS
used precision and that mismatch has bitten everyone at least once). PyMC ships a
convenience distribution, `pm.NormalMixture`, that takes the weights and the
per-component means and sigmas directly. Here is the whole model for $K = 2$, and
then I will dissect every line:

```python
K = 2
coords = {"cluster": range(K)}

# This is the from-scratch model, shown once to expose every line. §6 rebuilds the
# *same* model through the `make_gmm` factory (objects there are named gmm2_fit /
# gmm2_prior to keep the contexts straight); treat this block as the explainer.
with pm.Model(coords=coords) as gmm2:
    # Mixing weights on the simplex. Symmetric uniform prior.
    w = pm.Dirichlet("w", a=np.ones(K), dims="cluster")

    # Component means. The ORDERED transform breaks label-switching symmetry
    # (explained in detail in the next section). initval MUST be strictly increasing.
    mu = pm.Normal(
        "mu",
        mu=70.0, sigma=20.0,
        transform=pm.distributions.transforms.univariate_ordered,  # newer alias: transforms.ordered
        initval=[55.0, 80.0],   # sorted; one near each visible hump
        dims="cluster",
    )

    # Component standard deviations. Positive by construction.
    sigma = pm.HalfNormal("sigma", sigma=15.0, dims="cluster")

    # The marginalized mixture likelihood. No explicit z label appears!
    x_obs = pm.NormalMixture("x_obs", w=w, mu=mu, sigma=sigma, observed=x)

    idata2 = pm.sample(
        1000, tune=1000, chains=4,
        target_accept=0.9, random_seed=RANDOM_SEED,
    )
```

Line by line:

- **`coords={"cluster": range(K)}`** — named dimensions, so `az.summary` will label
  your two means `mu[0]` and `mu[1]` against a readable `cluster` coordinate instead
  of anonymous indices. This pays off enormously once $K$ grows.
- **`w = pm.Dirichlet("w", a=np.ones(K), dims="cluster")`** — the weights, with the
  symmetric prior from §3.1. `dims="cluster"` ties its length to $K$.
- **`mu = pm.Normal(...)`** — the component means. The prior `Normal(70, 20)` is
  centered in the bulk of the waiting times (which run ~43–96 minutes) and wide
  enough not to fight the data, but informative enough to keep the sampler sane.
  The two extra arguments — `transform=...univariate_ordered` and
  `initval=[55.0, 80.0]` — are the label-switching cure, covered next.
- **`sigma = pm.HalfNormal("sigma", sigma=15.0, dims="cluster")`** — per-component
  standard deviations. `HalfNormal` keeps them positive; a scale of 15 minutes is a
  weakly-informative guess at within-cluster spread.
- **`x_obs = pm.NormalMixture(...)`** — the likelihood. Notice what is *missing*:
  there is no `z`, no `Categorical`, no per-observation label anywhere. `NormalMixture`
  evaluates the marginalized density $\sum_k w_k\,\mathrm{Normal}(x_n\mid\mu_k,\sigma_k)$
  for you. That is the whole reason NUTS can sample this model, and §5 explains why.

> 🔧 **In practice — the equivalent `pm.Mixture` spelling.** `NormalMixture` is sugar
> for the general `pm.Mixture`, which takes a list of component distributions (or one
> batched distribution) via `comp_dists`. The components must be built with the
> **`.dist()` API** — *unregistered* distributions, not named model variables — and
> the mixture axis is the **last** axis. The two equivalent ways to write our $K=2$
> Gaussian mixture:
>
> ```python
> # (a) one batched component dist, vectorized over the last axis:
> comp = pm.Normal.dist(mu=mu, sigma=sigma, shape=(K,))
> x_obs = pm.Mixture("x_obs", w=w, comp_dists=comp, observed=x)
>
> # (b) a list of component dists (lets you mix DIFFERENT families):
> comps = [pm.Normal.dist(mu=mu[0], sigma=sigma[0]),
>          pm.StudentT.dist(nu=4, mu=mu[1], sigma=sigma[1])]
> x_obs = pm.Mixture("x_obs", w=w, comp_dists=comps, observed=x)
> ```
>
> Form (a) is the homogeneous case (`NormalMixture` is exactly this under the hood).
> Form (b) is the superpower: a mixture of a Normal "bulk" and a Student-$t$ "outlier"
> component, or a Poisson and a NegativeBinomial, etc. If you build components with
> `pm.Normal("c", ...)` (a *named* RV) instead of `pm.Normal.dist(...)`, PyMC raises an
> error about registered variables — that is the single most common `pm.Mixture` bug.

---

## 4. Identifiability and label switching: the trap, and the one-line cure

This is the most important section in the chapter. Read it twice.

### 4.1 Why the symmetry exists

Look again at the marginalized mixture likelihood:

$$
p(y \mid w, \mu, \sigma) \;=\; \prod_{n=1}^{N} \sum_{k=1}^{K} w_k \, \mathrm{Normal}(y_n \mid \mu_k, \sigma_k).
$$

Now ask: what happens if I **relabel** the components — swap component 1 and
component 2, carrying their weights, means, and sigmas along? Concretely, replace
$(w_1, \mu_1, \sigma_1) \leftrightarrow (w_2, \mu_2, \sigma_2)$. The sum over $k$ has
the *exact same terms*, just added in a different order. Addition is commutative.
**The likelihood is unchanged.**

This is the crux: the likelihood is **invariant to permutations of the component
labels**. There is nothing in the data that distinguishes "component 1 is the
short-wait hump and component 2 is the long-wait hump" from "component 1 is the
long-wait hump and component 2 is the short-wait hump." Both descriptions fit the
data identically. With $K$ components there are $K!$ such relabelings, and each one
is an *exact copy* of the posterior, reflected into a different corner of parameter
space.

> 🧮 **The math:** Let $\pi$ be any permutation of $\{1,\dots,K\}$ and write
> $\theta = (w, \mu, \sigma)$. Then for the permuted parameters
> $\theta^\pi = (w_{\pi(k)}, \mu_{\pi(k)}, \sigma_{\pi(k)})_k$ we have
> $p(y \mid \theta^\pi) = p(y \mid \theta)$ exactly. If the prior is also symmetric
> across components (and ours is — same prior on every $\mu_k$, same on every
> $\sigma_k$, symmetric Dirichlet), then the *posterior* inherits the symmetry too:
> $p(\theta^\pi \mid y) = p(\theta \mid y)$. The posterior literally has $K!$
> identical modes. This is a **non-identifiability**: the parameters are not
> uniquely determined by the data and prior, no matter how much data you collect.

> 💡 **Intuition:** Imagine two indistinguishable chairs at a table, "chair A" and
> "chair B." If I tell you only "one person sits left, one sits right," there is no
> fact of the matter about which chair is "A." Naming them A and B was *our* choice,
> not the world's. The mixture components are those chairs. The data knows there are
> two clusters; it does not know, and cannot know, which one we decided to call
> "number 1."

### 4.2 What it does to your chains — the symptoms

A symmetric posterior with $K!$ identical modes is poison for MCMC *if you intend
to interpret the component parameters individually*. Here is what you will see, and
you must learn to recognize it instantly:

- **Bimodal (multimodal) trace plots for `mu` and `w`.** One chain settles into the
  labeling "$\mu_0 \approx 54$, $\mu_1 \approx 80$"; another chain settles into the
  mirror labeling "$\mu_0 \approx 80$, $\mu_1 \approx 54$." Plotted together, `mu[0]`
  looks like it is wandering between two totally different values.
- **$\hat R$ enormous on `mu`, `sigma`, `w`** — often $\hat R \approx 1.5$–$2$. Recall
  from Chapter 7 the threshold $\hat R \le 1.01$ for a healthy parameter. Here $\hat R$
  blows up not because any chain failed to converge, but because different chains
  converged to *different (equivalent) labelings*. $\hat R$ compares between-chain to
  within-chain variance, and the between-chain variance is huge.
- **Even a single chain can "switch" mid-run**, hopping from one labeling to its
  mirror, giving a trace that looks like a telegraph signal.
- **Forest plots where the component intervals overlap or straddle both humps.**

> ⚠️ **Pitfall:** The deadliest version of this is the *silent* one. If your two
> humps are well separated and your chains happen to all land in the same labeling
> by luck, $\hat R$ might look fine — until you re-run with a different seed and half
> your "component 1 mean" estimates flip. Never trust per-component estimates from a
> mixture that has no identifiability constraint, even if this particular run looks
> clean. Build the constraint in from the start.

> 🩺 **Diagnostic — how to *see* label switching.** Run `az.plot_trace(idata,
> var_names=["mu"])` on a mixture fit *without* the ordered constraint. The left
> column (the marginal densities) for `mu[0]` will show **two separated peaks**, and
> the right column (the per-chain time series) will show chains sitting at different
> levels or jumping between them. `az.plot_rank(idata, var_names=["mu"])` is even
> sharper: healthy chains give uniform rank histograms, but under label switching
> you get wildly non-uniform, chain-dependent rank plots. And `az.summary(idata)`
> will report the $\hat R \approx 2$ that confirms it.

### 4.3 The fix: order the component means

The cure is beautifully simple. We do not actually care *which* hump we call
"component 0"; we only need a *rule* that picks one labeling consistently. So we
impose the rule **"the means are sorted in increasing order"**:

$$
\mu_1 < \mu_2 < \cdots < \mu_K.
$$

Out of the $K!$ equivalent labelings, exactly one satisfies this constraint. By
restricting the parameter space to that one ordering, we collapse the $K!$ modes
into a single mode. The data still determines *where* the humps are; we have only
fixed the bookkeeping of which is called first.

PyMC implements this with an **ordered transform** applied to the `mu` variable.
You already saw it in the model above; here is the line in isolation:

```python
mu = pm.Normal(
    "mu",
    mu=70.0, sigma=20.0,
    transform=pm.distributions.transforms.univariate_ordered,  # newer alias: transforms.ordered
    initval=[55.0, 80.0],   # MUST be strictly increasing
    dims="cluster",
)
```

What the transform does under the hood: it reparameterizes the constrained vector
$(\mu_1 < \mu_2 < \dots)$ into an *unconstrained* vector that NUTS can explore
freely. Concretely, it stores the first element directly and the *gaps* between
successive elements through a log map:
$$
y_1 = \mu_1, \qquad y_k = \log(\mu_k - \mu_{k-1}) \;\; (k > 1),
$$
an unconstrained vector $y \in \mathbb{R}^K$ that NUTS samples. The inverse
exponentiates the gaps and cumulatively sums — $\mu_1 = y_1$,
$\mu_k = \mu_{k-1} + e^{y_k}$ — and because each $e^{y_k} > 0$ the result is
*guaranteed* strictly increasing $\mu_1 < \mu_2 < \dots < \mu_K$. The
$\log|\text{Jacobian}|$ correction of that change of variables is added to the log
density automatically, so the prior stays correct on the constrained scale. You
write "ordered" and PyMC does the calculus. This is the same trick that lets
`HalfNormal` live on $(0, \infty)$ while NUTS samples on $(-\infty, \infty)$.

> ⚠️ **Pitfall — the `initval` must be sorted.** The ordered transform's starting
> point must already satisfy the constraint, strictly. If you pass
> `initval=[80, 55]` (descending) or `initval=[55, 55]` (tied), you get a
> `SamplingError: Initial evaluation of model at starting point is not finite` (or a
> transform error). Always pass strictly increasing initial values, ideally placed
> near where you expect the humps — here `[55, 80]`, one per visible cluster.

> 📜 **Citation/Origin & a caveat.** Stan bakes this in as a *type*: you declare
> `ordered[K] mu;` and it is ordered by construction (we will see the full Stan
> model in §5). PyMC achieves the same with a transform. **The honest caveat**,
> straight from the PyMC Discourse: the ordered constraint is the right default, but
> when components *genuinely overlap heavily* — their means are close relative to
> their spreads — forcing an order can carve up a smooth posterior awkwardly and
> *worsen* the geometry, occasionally introducing spurious peaks or divergences. In
> that regime the alternatives are (a) reduce $K$, (b) use stronger priors that pull
> the means apart, or (c) drop the constraint and do **post-hoc relabeling** of the
> draws. But order-the-means is what you reach for first, and for well-separated
> data like Old Faithful it works cleanly.

> 🔧 **In practice — which spelling?** Recent PyMC consolidated the ordered
> transforms: `univariate_ordered` and `multivariate_ordered` are now deprecated
> aliases that emit a `FutureWarning` and forward to a single `ordered` transform
> (class `pm.distributions.transforms.Ordered`, which has `ascending`/`positive`
> options). The current example gallery still shows `univariate_ordered`, so that is
> what I write here, with the newer `transforms.ordered` noted in a comment. If your
> installed version warns, switch the spelling to `pm.distributions.transforms.ordered`
> — the behavior is identical. *(Verify in the current PyMC docs which name is
> non-deprecated in your version.)*

---

## 5. Marginalizing the discrete label: why, and what it looks like in Stan

We have been quietly skating over something important. The generative story has a
discrete latent label $z_n$ per observation. But our PyMC model has no `z` in it at
all — and our chains ran with NUTS. How?

### 5.1 Why HMC/NUTS cannot touch a discrete parameter

Recall from Chapter 5 how Hamiltonian Monte Carlo works. It treats the negative log
posterior as a potential-energy surface, gives the particle momentum, and
integrates Hamilton's equations of motion with the leapfrog integrator to glide
along that surface. Every bit of that machinery needs **gradients** of the log
density with respect to the parameters. Gradients require the parameter to be
**continuous** — you have to be able to nudge it a tiny bit and ask "how did the log
density change?"

A discrete label $z_n \in \{1, \dots, K\}$ has no such notion. There is no "tiny
nudge" between component 1 and component 2; the density is a step function in $z_n$,
its gradient is zero almost everywhere and undefined at the jumps. HMC has nothing
to push against. **HMC/NUTS simply cannot sample discrete parameters.** (Stan
*forbids* declaring them outright; PyMC will fall back to a clunky compound step for
discrete variables, which mixes poorly and defeats the point of using NUTS.)

So what do we do with the discrete $z_n$? We **sum it out** — we marginalize it
analytically, turning the joint over $(y, z)$ into a likelihood over $y$ alone:

$$
p(y_n \mid w, \theta) \;=\; \sum_{z_n=1}^{K} p(z_n \mid w)\, p(y_n \mid z_n, \theta)
\;=\; \sum_{k=1}^{K} w_k \, f(y_n \mid \theta_k).
$$

That is *exactly* the mixture density from §1. The discrete label vanishes; what
remains is a smooth, differentiable function of the continuous parameters
$(w, \mu, \sigma)$. **Now NUTS has gradients and runs happily.** Marginalization is
not a clever optimization here — for HMC it is a *prerequisite*. And it is not even
expensive: the sum has only $K$ terms per observation.

> 💡 **Intuition:** "Marginalize the label" sounds heavy but it means something
> homely: instead of guessing which component each point came from and conditioning
> on the guess, we *average over all the possibilities*, weighted by how plausible
> each is. We refuse to commit to a label and integrate our ignorance away. That
> averaging is what makes the surface smooth.

> 🔧 **In practice — there is a bonus beyond "NUTS can run."** Even when you *could*
> sample the discrete $z$ (e.g. with Gibbs), marginalizing it first usually gives
> **dramatically better mixing and tail exploration**. Sampling $z$ creates a huge,
> rugged discrete space the chain must shuffle through one label-flip at a time;
> integrating it out replaces that with a low-dimensional continuous geometry NUTS
> traverses efficiently. The PyMC "Marginalized Gaussian Mixture" gallery example
> makes exactly this point. So even outside the HMC requirement: **marginalize when
> you can.**

### 5.2 The numerically stable form: `log_sum_exp`

We always work on the log scale (products of many small densities underflow to
zero). The log of the marginalized likelihood for one observation is

$$
\log p(y_n) = \log \sum_{k=1}^{K} \exp\!\big(\log w_k + \log f(y_n \mid \theta_k)\big).
$$

That "log of a sum of exponentials of logs" pattern is so common it has a name and
a numerically stable implementation: **`log_sum_exp`**. Done naively, the inner
$\exp$ can overflow or underflow; `log_sum_exp` factors out the maximum term first
($\log\sum_k e^{a_k} = a^\* + \log\sum_k e^{a_k - a^\*}$ with $a^\* = \max_k a_k$) so
the arithmetic stays in a safe range. PyMC uses this internally inside
`Mixture`/`NormalMixture`; in Stan you call it yourself.

### 5.3 Doing it by hand in Stan, so you see the gears

Here is the *same* Gaussian mixture, written in Stan via CmdStanPy. Stan makes the
marginalization explicit, which is exactly why it is so instructive — you can see
the `log_sum_exp` that PyMC hides:

```stan
// gaussian_mixture.stan
data {
  int<lower=1> N;          // number of observations
  int<lower=1> K;          // number of components
  vector[N] y;             // the data
}
parameters {
  simplex[K] theta;        // mixing weights, on the simplex by construction
  ordered[K] mu;           // ORDERED locations -> identifiability, native to Stan
  vector<lower=0>[K] sigma; // positive scales by construction
}
model {
  // priors
  mu ~ normal(70, 20);
  sigma ~ normal(0, 15);   // half-normal because of the <lower=0> constraint
  // theta ~ dirichlet(rep_vector(1, K));  // implicit uniform if omitted

  // the marginalized mixture likelihood, summed over the latent label
  vector[K] log_theta = log(theta);
  for (n in 1:N) {
    vector[K] lps = log_theta;                 // start with log w_k
    for (k in 1:K)
      lps[k] += normal_lpdf(y[n] | mu[k], sigma[k]);  // + log f(y_n | theta_k)
    target += log_sum_exp(lps);                // log Σ_k exp(...) -> add to log posterior
  }
}
```

Run it from Python:

```python
from cmdstanpy import CmdStanModel

model = CmdStanModel(stan_file="gaussian_mixture.stan")
fit = model.sample(
    data={"N": len(x), "K": 2, "y": x},
    seed=RANDOM_SEED, chains=4, iter_warmup=1000, iter_sampling=1000,
)
idata_stan = az.from_cmdstanpy(fit)   # same ArviZ InferenceData, same diagnostics
```

Walk through the `model` block, because it is the entire lesson:

- **`simplex[K] theta`** and **`ordered[K] mu`** are *constrained types*. Stan gives
  you the simplex and the ordering for free — no transform argument, no `initval`
  sorting. `ordered[K] mu` is Stan's native label-switching fix.
- **`target += log_sum_exp(lps)`** is the line that matters. `target` is Stan's
  accumulator for the log posterior. For each observation we build the vector `lps`
  of $\log w_k + \log\mathrm{Normal}(y_n \mid \mu_k, \sigma_k)$, then add its
  `log_sum_exp` to the running total. **That is the marginalization, by hand.** There
  is no `z` declared anywhere — *there cannot be*, because Stan refuses discrete
  parameters. You are forced to confront what PyMC quietly did for you.

> 🔧 **In practice — the two-component sugar.** For exactly $K = 2$ components Stan
> offers `target += log_mix(lambda, lpdf1, lpdf2)`, which is `log_sum_exp` of the two
> branches with weight `lambda`. It is the same computation, prettier for the binary
> case. PyMC's `NormalMixture` covers all $K$ uniformly, so there is no separate
> two-component shortcut on that side.

> 💡 **The big-picture takeaway.** PyMC and Stan compute the *identical* objective —
> the marginalized, ordered mixture log-likelihood. PyMC hides the
> sum-over-labels behind `Mixture`; Stan makes you type the `log_sum_exp`. Neither
> samples the discrete label, because neither *can* with HMC. Understanding the Stan
> version means you understand what `pm.Mixture` is and why it exists. This same
> "sum out the discrete latent so HMC can run" pattern returns in Chapter 16 for
> missing-data indicators, so the investment compounds.

---

## 6. Worked end-to-end example: a $K$-component GMM on Old Faithful

Now we run the full workflow on the geyser waiting times — prior predictive check,
fit, diagnostics, posterior predictive check, choosing $K$ with LOO, and finally
recovering cluster membership. This is the section to keep open next to your editor.

### 6.1 A reusable model builder

We parameterize the build by $K$ so we can compare component counts later:

```python
import numpy as np
import seaborn as sns
import pymc as pm
import arviz as az

RANDOM_SEED = 8927
geyser = sns.load_dataset("geyser")
x = geyser["waiting"].to_numpy().astype(float)

def make_gmm(x, K, seed=RANDOM_SEED):
    # spread K starting means evenly across the data range, strictly increasing
    init_means = np.linspace(x.min() + 5, x.max() - 5, K)
    coords = {"cluster": range(K)}
    with pm.Model(coords=coords) as model:
        w = pm.Dirichlet("w", a=np.ones(K), dims="cluster")
        mu = pm.Normal(
            "mu", mu=70.0, sigma=20.0,
            transform=pm.distributions.transforms.univariate_ordered,  # newer alias: transforms.ordered
            initval=init_means,            # strictly increasing -> required
            dims="cluster",
        )
        sigma = pm.HalfNormal("sigma", sigma=15.0, dims="cluster")
        pm.NormalMixture("x_obs", w=w, mu=mu, sigma=sigma, observed=x)
    return model
```

### 6.2 Prior predictive check — does the model *a priori* produce geyser-like data?

Before fitting, simulate from the prior and confirm the implied data is plausible.
This is the workflow discipline from Chapter 8: never fit a model whose prior
predictive is absurd.

```python
with make_gmm(x, K=2) as gmm2_prior:
    # v5: the argument is `draws` (it was `samples` in PyMC3 / very early v5).
    prior_pred = pm.sample_prior_predictive(draws=200, random_seed=RANDOM_SEED)

az.plot_ppc(prior_pred, group="prior", num_pp_samples=100)
```

> 🩺 **Diagnostic:** `az.plot_ppc(..., group="prior")` overlays many simulated
> datasets (thin lines) against the real data's KDE (thick line). You are *not*
> looking for a perfect match — the prior hasn't seen the data. You want the
> simulated waiting times to stay on a sane order of magnitude (tens of minutes, not
> $-500$ or $10^6$), to be able to produce bimodality, and to comfortably *cover* the
> observed range (~43–96 min) without being so tight it excludes it. With
> `Normal(70, 20)` means and `HalfNormal(15)` sigmas the bulk of the prior draws sit
> over the data; the tails will occasionally wander somewhat below 0 or well past 140
> — that is *fine*, and even desirable: a weakly-informative prior should over-cover
> rather than fence the data in. What you do *not* want is the absurd-scale or
> data-excluding behavior. Good — these priors are weakly informative, not
> accidentally dogmatic.

### 6.3 Fit and check diagnostics

```python
# `make_gmm` returns the Model object, which `as gmm2_fit` binds; this is the
# model we will reuse for posterior predictive in §6.4. (It is a *fresh* model,
# distinct from the standalone `gmm2` built inline in §3.2 — don't mix them up.)
with make_gmm(x, K=2) as gmm2_fit:
    idata2 = pm.sample(
        1000, tune=1000, chains=4,
        target_accept=0.9, random_seed=RANDOM_SEED,
    )

az.summary(idata2, var_names=["w", "mu", "sigma"])
```

A healthy run produces a summary table like this (your exact numbers will vary
slightly by seed and version):

```
           mean     sd   hdi_3%  hdi_97%  ess_bulk  ess_tail  r_hat
w[0]      0.362  0.030    0.307    0.418      3100      2800   1.00
w[1]      0.638  0.030    0.582    0.693      3100      2800   1.00
mu[0]    54.62   0.69    53.33    55.91      3400      2900   1.00
mu[1]    80.09   0.50    79.16    81.04      3600      3000   1.00
sigma[0]  5.90   0.55     4.90     6.93      2700      2600   1.00
sigma[1]  5.87   0.40     5.13     6.63      2900      2700   1.00
```

> 🩺 **Diagnostic — read this table the way Chapter 7 taught you.**
> - **`r_hat` = 1.00 everywhere** ($\le 1.01$): the chains agree. Crucially, because we
>   ordered the means, there is *no* label switching, so this is real convergence and
>   not a lucky labeling. Re-run with another seed and `mu[0]` stays the short hump.
> - **`ess_bulk` and `ess_tail` in the thousands** ($\gg 400$): plenty of effective
>   samples for stable point and interval estimates.
> - **Zero divergences** (check the sampler warnings, or `idata2.sample_stats["diverging"].sum()`).
> - **The estimates make physical sense:** a short-wait component at $\mu_0 \approx 55$
>   min carrying $w_0 \approx 36\%$ of eruptions, and a long-wait component at
>   $\mu_1 \approx 80$ min carrying $w_1 \approx 64\%$. The model recovered the two
>   regimes you saw in the histogram, and quantified them with uncertainty.

Always look at the plots too:

```python
az.plot_trace(idata2, var_names=["w", "mu", "sigma"])
az.plot_forest(idata2, var_names=["mu"], combined=True)
```

> 🩺 **Diagnostic:** With the ordered constraint in place, `az.plot_trace` should
> show, for each `mu` component, a **single** clean unimodal density (left column)
> and four fuzzy-caterpillar chains overlapping at the same level (right column). If
> you ever see *two* peaks in the left column for a single `mu[k]`, label switching
> has leaked back in — go back and check your transform and `initval`. The forest
> plot shows the two means cleanly separated with non-overlapping intervals, the
> visual signature of two genuinely distinct clusters.

> 🧪 **Try the broken version once.** Build the same model *without* the
> `transform=...` and `initval=...` arguments and sample it. Watch `r_hat` on `mu`
> jump toward 2, watch `az.plot_trace` show two peaks per mean, watch chains sit at
> different levels. Then add the two arguments back and watch it heal. Seeing the
> disease once is worth a thousand warnings.

### 6.4 Posterior predictive check — does the fitted mixture reproduce the data?

```python
# Posterior predictive must run inside the *same* model context that produced
# idata2 — here that is gmm2_fit from §6.3 (not the §3.2 `gmm2`).
with gmm2_fit:
    pm.sample_posterior_predictive(idata2, extend_inferencedata=True,
                                   random_seed=RANDOM_SEED)

az.plot_ppc(idata2, num_pp_samples=200)
```

> 🩺 **Diagnostic:** `az.plot_ppc` overlays the KDE of many *posterior-predictive*
> datasets against the observed KDE. For a good two-component fit, the cloud of
> simulated densities should hug the real bimodal shape — *both* humps and the
> valley between them reproduced. This is the payoff plot: it shows the mixture
> captured the structure a single Normal could not. If you fit $K=1$ (a plain
> Normal) and ran this, you would see a single broad simulated hump sitting over the
> empty valley while the data has two peaks — a textbook PPC failure that screams
> "you need more components."

### 6.5 Choosing the number of components $K$ with LOO

How many components? The eyes said two for Old Faithful, but in general you should
let predictive performance arbitrate. We use **PSIS-LOO** from Chapter 11: fit
candidate models, then compare expected log predictive density (ELPD) with
`az.compare`.

```python
idatas = {}
for K in (1, 2, 3, 4):
    with make_gmm(x, K=K) as m:
        idatas[f"K={K}"] = pm.sample(
            1000, tune=1000, chains=4, target_accept=0.95,
            idata_kwargs={"log_likelihood": True},   # LOO needs pointwise log-lik
            random_seed=RANDOM_SEED,
        )

comparison = az.compare(idatas)      # ranks by ELPD-LOO
print(comparison)
az.plot_compare(comparison)
```

A representative comparison table looks like this (exact numbers vary by seed and
version; `rank 0` is the best model):

```
      rank   elpd_loo   p_loo   elpd_diff     se    dse  warning
K=2      0   -1035.8     4.9       0.00     11.6   0.00    False
K=3      1   -1036.4     6.7       0.62     11.7   1.05    False
K=4      2   -1037.1     8.6       1.30     11.8   1.62    False
K=1      3   -1101.5     2.0      65.70     13.4   9.80    False
```

> 🩺 **Diagnostic — reading `az.compare`.** The table ranks models by ELPD-LOO
> (higher is better predictive accuracy). The top row is the best model; `elpd_diff`
> and `dse` (standard error of the difference) tell you whether the runners-up are
> *meaningfully* worse — a difference within a couple of standard errors is a tie.
> For Old Faithful you will typically find $K=2$ decisively beats $K=1$ (in the table
> above the $K=1$ `elpd_diff \approx 66` dwarfs its `dse \approx 10` — the single
> Normal can't do the valley), while $K=3$ and $K=4$ sit within ~1–2 `dse` of $K=2$
> (`elpd_diff` of $0.62$ and $1.30$) — a statistical tie, so you prefer the simplest,
> $K=2$. Extra components just split a real cluster into near-duplicates that don't
> help prediction. **Watch the `warning` / `pareto_k` column:** as $K$
> grows past what the data supports, components become weakly identified, the
> Pareto-$\hat k$ values climb above the **0.7** trust threshold, and the LOO
> estimate itself becomes unreliable. That climbing $\hat k$ is itself a signal that
> you are over-componenting.

> ⚠️ **Pitfall — `log_likelihood=True` is not automatic for mixtures.** LOO needs the
> *pointwise* log-likelihood stored in the InferenceData. Pass
> `idata_kwargs={"log_likelihood": True}` to `pm.sample` (shown above) or compute it
> afterward with `pm.compute_log_likelihood(idata, model=m)`. Forgetting this is the
> most common reason `az.loo` / `az.compare` raises "log likelihood not found."

> 🔧 **In practice — the deeper question.** "How many components?" is genuinely hard,
> because mixtures are happy to add near-empty or near-duplicate components without
> hurting the likelihood much. Three honest answers: (1) use LOO and prefer the
> simplest $K$ that isn't decisively beaten, as above; (2) use a **sparse Dirichlet**
> ($\alpha < 1$) to let the model switch surplus components off; (3) stop fixing $K$
> at all and go **nonparametric** — the Dirichlet-process mixture of Chapter 14 puts
> a prior over *infinitely* many components and lets the data decide how many are
> occupied. For density estimation that is often the cleanest path.

---

## 7. Recovering cluster membership: posterior responsibilities

We marginalized the label $z_n$ away to make NUTS work. But often we *wanted* the
label — "which cluster does eruption $n$ belong to?" is the whole point of
clustering. Good news: we threw nothing away. We can recover a full posterior over
each $z_n$ **after** fitting, by Bayes' rule applied to the label.

Given a single posterior draw of $(w, \mu, \sigma)$, the probability that
observation $n$ came from component $k$ — its **responsibility** — is

$$
r_{nk} \;=\; p(z_n = k \mid y_n, w, \mu, \sigma)
\;=\; \frac{w_k \, \mathrm{Normal}(y_n \mid \mu_k, \sigma_k)}
{\sum_{j=1}^{K} w_j \, \mathrm{Normal}(y_n \mid \mu_j, \sigma_j)}.
$$

> 🧮 **The math:** This is just Bayes' theorem for the discrete label. The numerator
> is "prior probability of component $k$ times likelihood of $y_n$ under component
> $k$"; the denominator is the marginal likelihood of $y_n$ (the same mixture density
> we have been summing all chapter) and renormalizes the responsibilities to sum to
> one across $k$. The word "responsibility" is the EM-algorithm term: it is how much
> component $k$ is "responsible for" explaining point $n$.

The Bayesian twist is that we have a *whole posterior* of $(w, \mu, \sigma)$, not a
point estimate. So we compute $r_{nk}$ for **every posterior draw** and average,
giving a posterior-mean responsibility that honestly reflects parameter
uncertainty:

```python
import numpy as np
from scipy.stats import norm

post = idata2.posterior
# stack chains+draws into one sample axis
w_s     = post["w"].stack(sample=("chain", "draw")).values      # (K, S)
mu_s    = post["mu"].stack(sample=("chain", "draw")).values     # (K, S)
sigma_s = post["sigma"].stack(sample=("chain", "draw")).values  # (K, S)

K, S = w_s.shape
N = x.shape[0]

# weighted component densities: shape (N, K, S)
comp = (
    w_s[None, :, :]
    * norm.pdf(x[:, None, None], loc=mu_s[None, :, :], scale=sigma_s[None, :, :])
)
resp = comp / comp.sum(axis=1, keepdims=True)   # normalize over K -> (N, K, S)

resp_mean = resp.mean(axis=2)        # (N, K): posterior-mean responsibility
hard_label = resp_mean.argmax(axis=1)  # (N,): MAP cluster, if you must pick one
```

Now you can do genuinely Bayesian clustering:

```python
# How confident is the assignment for each point?
confidence = resp_mean.max(axis=1)

# Points the model is UNSURE about live in the valley between the humps:
unsure = x[confidence < 0.8]
print(f"{len(unsure)} eruptions have <80% cluster confidence; "
      f"their waiting times: {np.sort(unsure)}")
```

> 💡 **Intuition:** This is the difference between Bayesian clustering and k-means.
> K-means hands every point a hard, overconfident label. Our `resp_mean` says, for an
> eruption that waited 54 minutes, "97% chance short-wait cluster"; for one that
> waited 68 minutes — right in the valley — it might say "55% short, 45% long," an
> honest *I'm-not-sure*. The points with low `confidence` are exactly the ones near
> the overlap of the two humps, where the data genuinely cannot tell. Reporting that
> uncertainty is the entire reason we went Bayesian.

> 🩺 **Diagnostic — visualize the assignment.** Color the histogram of `x` by
> `hard_label`, or better, plot each point's `resp_mean[:, 1]` (probability of the
> long-wait cluster) against its waiting time:
>
> ```python
> import matplotlib.pyplot as plt
> order = np.argsort(x)
> plt.scatter(x[order], resp_mean[order, 1], s=10)
> plt.xlabel("waiting (min)"); plt.ylabel("P(long-wait cluster)")
> ```
>
> You will see a smooth S-curve: near 0 for short waits, near 1 for long waits, and
> a soft transition through the valley. That S-curve *is* the posterior uncertainty
> about membership, and it is far more informative than a hard cut at some threshold.

> ⚠️ **Pitfall — responsibilities require identified labels.** Computing $r_{nk}$ only
> means something if "component 0" refers consistently to the same cluster across
> draws. That is *another* reason the ordered constraint matters: without it, half
> your draws would have components 0 and 1 swapped, and averaging responsibilities
> across draws would blend "short" and "long" into mush. Order first, then recover
> membership.

---

## 8. Zero-inflated and hurdle models: mixtures for excess zeros

Mixtures aren't only for clustering continuous data. One of their most common
real-world uses is for **counts with too many zeros** — and these models are just a
two-component mixture in disguise, with one component being "a spike at zero."

### 8.1 The problem: a Poisson can't make enough zeros

Suppose you count something — manuscripts a monk copies per day, fish caught per
park visitor, insurance claims per policy. You fit a Poisson, and the
posterior-predictive check shows the model produces far fewer zeros than your data
has. The Poisson's zero probability is pinned to its mean: $P(Y=0) = e^{-\mu}$. If
the mean is 3, that's only a 5% chance of a zero — but maybe 30% of your
observations are zero. The Poisson literally cannot make that many zeros without
dragging its mean down to absurdity, which then ruins the fit for the positive
counts.

Why so many zeros? Usually because **two different processes** generate them:

- **Structural zeros:** the count-producing process was *never switched on* for this
  observation. A park visitor who doesn't fish catches zero fish — not because
  fishing was unlucky, but because they weren't fishing. A monk who took the day off
  copies zero manuscripts.
- **Sampling zeros:** the process *was* on, but happened to produce a zero. A visitor
  who fished all day and caught nothing. A monk who worked but finished nothing.

A plain Poisson sees only the second kind. A **zero-inflated** model adds a latent
switch for the first kind — and that switch is exactly a mixture label.

### 8.2 Zero-inflated Poisson (ZIP)

The generative story has a latent Bernoulli gate per observation: with probability
$\psi$ the count process is *active* and we draw a Poisson; with probability
$1 - \psi$ we are in the structural-zero state and the count is forced to 0.

$$
P(Y = 0) = (1 - \psi) + \psi\, e^{-\mu}, \qquad
P(Y = y) = \psi\, \frac{\mu^y e^{-\mu}}{y!} \quad (y \ge 1).
$$

> 🧮 **The math:** Read the zero probability carefully — zeros come from **two
> sources**, and we add them: the structural-zero branch contributes $1-\psi$, and
> the active branch *also* contributes its own Poisson zero $\psi e^{-\mu}$. A
> positive count can only come from the active branch, hence the lone $\psi$ factor
> on $P(Y=y)$ for $y \ge 1$. This is literally a two-component mixture: component A
> is a point mass at 0 (weight $1-\psi$), component B is a Poisson (weight $\psi$).

PyMC ships this as a first-class distribution. **Mind the convention:** PyMC's `psi`
is the probability the count process is **active** (the Poisson branch), so $1-\psi$
is the extra structural-zero mass.

```python
# Synthetic excess-zero counts with known truth (recovery demo)
rng = np.random.default_rng(RANDOM_SEED)
psi_true, lam_true, N = 0.65, 4.0, 500      # 65% active, 35% structural zeros
active = rng.random(N) < psi_true
counts = np.where(active, rng.poisson(lam_true, N), 0)

with pm.Model() as zip_model:
    psi = pm.Beta("psi", 2, 2)              # prob the count process is active
    mu  = pm.Exponential("mu", lam=1 / 5)   # rate 1/5 => prior mean 5 (Exponential mean = 1/lam)
    y = pm.ZeroInflatedPoisson("y", psi=psi, mu=mu, observed=counts)
    idata_zip = pm.sample(1000, tune=1000, chains=4,
                          target_accept=0.9, random_seed=RANDOM_SEED)

az.summary(idata_zip, var_names=["psi", "mu"])
```

> 🩺 **Diagnostic:** With 500 synthetic points the posterior should recover
> `psi ≈ 0.65` and `mu ≈ 4.0` — the true values — with tight intervals and `r_hat`
> $= 1.00$. Recovering known truth from synthetic data is the cleanest possible
> sanity check that you wired the model up correctly; do this whenever you meet a new
> likelihood. PyMC also exposes `pm.ZeroInflatedNegativeBinomial` (for excess zeros
> *and* over-dispersion) and `pm.ZeroInflatedBinomial`.

### 8.3 Hurdle models: the cleaner two-part story

A **hurdle** model handles excess zeros differently, and the distinction is a
favorite exam question, so let me nail it. In a hurdle model there is a clean
two-part split:

1. A Bernoulli "hurdle" decides **zero vs positive**.
2. If positive, the count comes from a **zero-truncated** count distribution — one
   that *can never emit 0*.

$$
P(Y = 0) = 1 - \psi, \qquad
P(Y = y) = \psi\, \frac{\mathrm{Poisson}(y \mid \mu)}{1 - e^{-\mu}} \quad (y \ge 1).
$$

> 🧮 **The math — ZIP vs Hurdle, the one table to remember.**
>
> | | Where can a zero come from? | Can the count process emit 0? |
> |---|---|---|
> | **Zero-inflated** | structural branch **OR** the count branch | **Yes** |
> | **Hurdle** | only the gate (the "below-the-hurdle" branch) | **No** (zero-truncated) |
>
> ZIP says: "zeros are over-represented because there's an *extra* always-zero
> process on top of a normal count process that itself sometimes makes zeros."
> Hurdle says: "getting *any* count at all is one decision; *how big* the count is,
> given it cleared the hurdle, is a separate decision, and a cleared hurdle means
> strictly positive." Use **hurdle** when a zero is categorically different from a
> positive count (you fished or you didn't); use **zero-inflated** when zeros could
> plausibly arise both structurally and by chance.

PyMC has hurdle distributions as first-class citizens too, with the *same* `psi`
convention (probability of the above-the-hurdle / positive branch):

```python
with pm.Model() as hurdle_model:
    psi = pm.Beta("psi", 2, 2)               # P(clears the hurdle -> positive)
    mu  = pm.Exponential("mu", lam=1 / 5)    # rate 1/5 => prior mean 5; rate of the zero-truncated Poisson
    y = pm.HurdlePoisson("y", psi=psi, mu=mu, observed=counts)
    idata_hurdle = pm.sample(1000, tune=1000, chains=4,
                             target_accept=0.9, random_seed=RANDOM_SEED)
```

`pm.HurdleNegativeBinomial` and the continuous `pm.HurdleGamma` (a point mass at 0
glued to a positive Gamma — perfect for, say, rainfall or insurance payouts that are
exactly-zero-or-positive) round out the family.

> 🔧 **In practice — which one, and how to choose.** Fit both, then let an
> **`az.compare` LOO** comparison and a **posterior predictive check on the zero
> count** decide. The PPC question is concrete: simulate datasets from each fitted
> model and compare the *number of zeros* to your data's number of zeros. Whichever
> family reproduces the zero mass *and* the positive-count shape wins. Both $\psi$
> and $\mu$ can be made functions of covariates (a logistic regression on the gate,
> a log-linear regression on the rate) — that is the zero-inflated/hurdle **GLM**,
> and it slots straight into the Chapter 9 machinery.

> ⚠️ **Pitfall — "my ZIP fit looks identical to a plain Poisson."** If the estimated
> `psi` is near 1, the data didn't actually need the structural-zero branch — fine,
> the model told you the truth. But if you *expected* excess zeros and `psi → 1`
> anyway, suspect you confused ZIP and hurdle, or your zeros are sampling zeros a
> bigger `mu` would explain, or you mislabeled the `psi` convention. Re-read §8.2 vs
> §8.3 and check which branch your `psi` controls.

---

## ⚠️ Common errors & how to fix them

| Symptom | Cause | Fix |
|---|---|---|
| `SamplingError: Initial evaluation ... not finite` on a mixture | `initval` for the ordered means is not strictly increasing (or a scale init is invalid) | Pass strictly increasing `initval` (e.g. `[55, 80]`); ensure scales use `HalfNormal`/`Exponential`; check Dirichlet `a > 0` |
| Bimodal trace plots, $\hat R \approx 2$ on `mu`/`w`, components "swap" between chains or re-runs | Label switching — the likelihood is invariant to relabeling, giving $K!$ identical modes | Add `transform=pm.distributions.transforms.univariate_ordered` (or `ordered`) on the component means; in Stan declare `ordered[K] mu` |
| Ordered transform makes convergence *worse*, adds spurious peaks or divergences | Components genuinely overlap; forcing an order carves up a smooth posterior badly | Reduce $K$; use priors that separate the means; or drop the constraint and relabel draws post hoc (see PyMC Discourse) |
| `pm.Mixture` raises about registered/named variables | Built `comp_dists` from named RVs (`pm.Normal("c", ...)`) instead of unregistered dists | Use the **`.dist()` API**: `pm.Normal.dist(mu=..., sigma=...)` |
| `w` and components misalign / shape error in `pm.Mixture` | The mixture axis must be the **last** axis; `w` length ≠ component axis | Build a batched component with `shape=(K,)` and a matching `Dirichlet(a=np.ones(K))`; verify broadcasting |
| `az.loo` / `az.compare` raises "log likelihood not found" | Pointwise log-likelihood wasn't stored | Pass `idata_kwargs={"log_likelihood": True}` to `pm.sample`, or call `pm.compute_log_likelihood(idata, model=m)` afterward |
| Climbing Pareto-$\hat k$ (> 0.7) as you raise $K$ | Extra components are weakly identified; LOO importance weights become unstable | Treat high $\hat k$ as evidence of over-componenting; prefer the smaller $K$; or use moment-matching / refit-LOO; or go nonparametric (Ch 14) |
| ZIP fit looks identical to a plain Poisson | Confused ZIP (zeros from 2 sources) with hurdle (clean gate + truncated count), or the data has no structural zeros | Pick the right family; remember PyMC `psi` = prob of the *active / above-hurdle* branch; compare via LOO + zero-count PPC |
| Responsibilities average into mush across draws | Labels not identified, so "component 0" means different clusters in different draws | Impose the ordered constraint *before* computing responsibilities; then `r_{nk}` is consistent across draws |
| `FutureWarning` about `univariate_ordered` | Your PyMC version deprecated the alias in favor of the consolidated `ordered` | Switch the spelling to `pm.distributions.transforms.ordered`; behavior is identical |
| Discrete `z` in the model + NUTS mixes terribly / errors | You declared an explicit categorical latent label that HMC can't differentiate | Don't sample the label — use `pm.Mixture`/`NormalMixture`, which marginalizes it; in Stan, `log_sum_exp` it out |

---

## 🧪 Exercises

1. **(Conceptual — feel the symmetry.)** Write out, for a $K=2$ Gaussian mixture, the
   full marginalized log-likelihood for a single observation. Now swap
   $(w_1,\mu_1,\sigma_1) \leftrightarrow (w_2,\mu_2,\sigma_2)$ symbolically and show the
   expression is unchanged. In one sentence, explain why this guarantees $\hat R$ will
   blow up on `mu` if you sample without an ordering constraint. *Hint: $\hat R$
   compares between-chain to within-chain variance.*

2. **(Coding — see the disease, then cure it.)** Fit the Old Faithful $K=2$ GMM
   **twice**: once without the ordered transform and `initval`, once with them. For
   each, report `az.summary` (focus on `r_hat` for `mu`) and show `az.plot_trace`
   and `az.plot_rank` for `mu`. Describe exactly how the broken run's trace and rank
   plots differ from the healthy one's. *Hint: look for two peaks in the marginal
   density and chains sitting at different levels.*

3. **(Coding — choose $K$ honestly.)** Run the $K \in \{1,2,3,4\}$ LOO comparison from
   §6.5 on the geyser **duration** column (not waiting). Report the `az.compare`
   table and the Pareto-$\hat k$ warnings. Which $K$ wins, and is the runner-up within
   one or two standard errors (`dse`) of the best? At what $K$ do the $\hat k$ values
   start crossing 0.7, and what does that tell you? *Hint: duration is also bimodal,
   but the separation differs.*

4. **(Coding — heterogeneous mixture.)** Using the general `pm.Mixture` with a
   **list** `comp_dists`, fit a two-component mixture of a `Normal` "bulk" and a
   `StudentT(nu=3)` "outlier" component to a dataset you contaminate with a few
   extreme points. Show that the responsibilities flag the contaminants as belonging
   to the heavy-tailed component. *Hint: this is robust modeling via an explicit
   outlier component, a cousin of the Student-$t$ likelihood from Chapter 9.*

5. **(Coding — ZIP vs hurdle showdown.)** Generate count data from a *true* hurdle
   process (gate, then zero-truncated Poisson). Fit **both** `pm.ZeroInflatedPoisson`
   and `pm.HurdlePoisson`. Compare them with `az.compare`, and do a posterior
   predictive check specifically on the **number of zeros** and the **count of ones**.
   Which model wins, and why does the zero-truncation of the hurdle matter for the
   count of ones? *Hint: a ZIP's active branch can still emit zeros; a hurdle's cannot
   — that changes the implied number of small positive counts.*

6. **(Conceptual / stretch — Bayesian vs k-means.)** Take the geyser responsibilities
   `resp_mean` from §7 and the hard labels from k-means (`sklearn.cluster.KMeans`,
   `n_clusters=2`). Identify the eruptions where the two methods *disagree* or where the
   Bayesian model reports `confidence < 0.8` (the same arbitrary cutoff §7 used — feel
   free to pick your own). Argue, with reference to the histogram's
   valley, why the Bayesian uncertainty is the more honest summary. *Hint: k-means
   draws a hard line; the mixture reports a probability.*

---

## 📚 Resources & further reading

- **PyMC — Gaussian Mixture Model (example gallery):**
  https://www.pymc.io/projects/examples/en/latest/mixture_models/gaussian_mixture_model.html
  — the canonical `pm.NormalMixture` + `univariate_ordered` + `Dirichlet` skeleton this
  chapter's model is built on.
- **PyMC — Marginalized Gaussian Mixture Model:**
  https://www.pymc.io/projects/examples/en/latest/mixture_models/marginalized_gaussian_mixture_model.html
  — the "why marginalize the label" narrative (mixing and tail exploration). Note its code
  is older PyMC3-flavored; the idea is what matters.
- **PyMC — Automatic marginalization of discrete variables (how-to):**
  https://www.pymc.io/projects/examples/en/latest/howto/marginalizing-models.html
  — for marginalizing discrete latents *beyond* `pm.Mixture` (the `pm.MarginalModel` machinery).
- **PyMC API — `pm.Mixture`:**
  https://www.pymc.io/projects/docs/en/latest/api/distributions/generated/pymc.Mixture.html
  — `w`/`comp_dists`, the `.dist()` requirement, list-vs-batched component forms.
- **PyMC API — `pm.NormalMixture`:**
  https://www.pymc.io/projects/docs/en/latest/api/distributions/generated/pymc.NormalMixture.html
  — per-component `mu`/`sigma`/`tau` semantics.
- **PyMC API — `pm.HurdlePoisson` & `pm.ZeroInflatedPoisson`:**
  https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.HurdlePoisson.html
  and
  https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.ZeroInflatedPoisson.html
  — exact PMFs and the `psi` convention; read the two pages side by side to settle the
  ZIP-vs-hurdle distinction.
- **PyMC API — `transforms.Ordered`:**
  https://www.pymc.io/projects/docs/en/latest/api/distributions/generated/pymc.distributions.transforms.Ordered.html
  and the source
  https://github.com/pymc-devs/pymc/blob/main/pymc/distributions/transforms.py
  — confirms the `univariate_ordered` → `ordered` consolidation; cite for the "which spelling?" question.
- **Stan User's Guide — Finite Mixtures:**
  https://mc-stan.org/docs/stan-users-guide/finite-mixtures.html
  — `ordered[K] mu`, `simplex[K] theta`, `log_sum_exp` / `log_mix`. The source for §5's Stan box.
- **Stan User's Guide — Latent Discrete Parameters:**
  https://mc-stan.org/docs/stan-users-guide/latent-discrete.html
  — why discrete parameters must be summed out under HMC; the broader marginalization context.
- **Martin, *Bayesian Analysis with Python* (3rd ed) — Mixture Models chapter:**
  https://github.com/aloctavodia/BAP
  — finite mixtures, non-identifiability, ZIP, and DP mixtures in idiomatic PyMC. This is the
  book-of-record for this chapter (the *Bayesian Modeling and Computation in Python* book has no
  standalone mixtures chapter).
- **McElreath, *Statistical Rethinking* 2e — Ch 12 "Monsters and Mixtures":**
  https://github.com/rmcelreath/rethinking (free lecture videos linked there)
  — zero-inflated Poisson (the monks), over-dispersion, and the latent-switch framing.
- **PyMC Discourse — ordered transform & label switching (real failure modes):**
  https://discourse.pymc.io/t/why-does-transform-pm-distributions-transforms-ordered-lead-to-worse-convergence/7944
  and https://discourse.pymc.io/t/issues-with-multivariate-gmm-model-and-label-switching/14290
  — when ordering helps, when it hurts, and what to do instead.
- **Gelman, Carlin, Stern, Dunson, Vehtari, Rubin, *Bayesian Data Analysis* (BDA3):**
  free PDF online — Ch 22 on finite mixtures and the identifiability discussion; the reference text.
- **Betancourt, "Identifying Bayesian Mixture Models":**
  https://betanalpha.github.io/assets/case_studies/identifying_mixture_models.html
  (index: https://betanalpha.github.io/writing/) — a rigorous case study on mixture
  non-identifiability, the label-switching geometry, and principled fixes. Read this
  once you're comfortable with §4.

---

## ➡️ What's next

You can now build, fit, diagnose, and interpret a *finite* mixture — you choose $K$,
order the means, and let LOO referee how many components the data supports. But that
nagging question, "how many components, really?", never fully goes away: the data
may want three clusters here and seven there, and fixing $K$ in advance is a
straitjacket. **Chapter 14 — Nonparametric & Flexible Models** removes the
straitjacket. We put a prior over *infinitely many* components via the
**Dirichlet-process mixture** (built with the stick-breaking construction you'll
recognize as a cousin of the Dirichlet weights here) and let the data decide how
many clusters are actually occupied. The same chapter zooms out to the broader
theme this one hinted at — *flexible function approximation without committing to a
parametric form* — covering **Bayesian splines** and **BART**, and tying all of it
back to the Gaussian processes of Chapter 12. Splines, GPs, BART, and infinite
mixtures turn out to be four answers to one question: *what do you do when you
refuse to commit to a fixed parametric shape?* Let's go meet the infinite.
