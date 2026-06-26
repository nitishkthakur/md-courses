# Chapter 02 — Priors I: Concepts & the Common-Prior Catalog

### **What a prior actually encodes, why "no prior" is a fiction, and the field guide to every distribution you'll reach for**

> *A note from me to you.*
>
> *Of everything in Bayesian modeling, the prior is the part that makes newcomers nervous and the part that makes experts shrug. The nervousness comes from a misunderstanding: people imagine the prior as a thumb on the scale, a way to smuggle your wishes into the answer. The shrug comes from experience: once you've fit a few hundred models, you realize the prior is just another part of your generative story — no more arbitrary than your choice of likelihood, and far less consequential than people fear once you have a sensible amount of data. But "far less consequential" is not "never consequential." Priors quietly cause some of the nastiest bugs in applied Bayes: variance parameters that collapse to zero, hierarchical models that won't sample, logistic regressions that explode under separation, prior predictive distributions that put a person's height at negative four meters. Almost all of those bugs trace back to someone treating "flat" as a synonym for "neutral," or reaching for a conjugate prior because the algebra was pretty rather than because it described their beliefs.*
>
> *This chapter is the definitive reference on priors-the-objects: what a prior is, the difference between flat and uninformative (they are not the same, and confusing them will hurt you), the conjugate families with their update formulas worked out in actual numbers, the ridge/lasso connection that ties everything back to the regularization you already know from ML, and then a long, careful field guide to every distribution you'll ever put on a parameter — with the one detail that catches everyone: PyMC parameterizes by standard deviation, not precision. The companion skill — how to choose priors, build them, and check them with prior predictive simulation — is Chapter 03. Here we build the vocabulary and the catalog. Read it like a reference you'll return to, because you will.*

## What you'll be able to do after this chapter

- Explain precisely what a prior **encodes** and why every Bayesian analysis has one, even when the modeler pretends it doesn't.
- Distinguish **flat**, **improper**, **proper**, **weakly informative**, **informative**, and **regularizing** priors — and say why "flat ≠ uninformative."
- Derive and *compute with* the three workhorse **conjugate** updates: Beta–Binomial, Gamma–Poisson, and Normal–Normal, reading the posterior mean as a convex combination of prior and data.
- Show that **ridge regression is MAP under a Gaussian prior** and **lasso is MAP under a Laplace prior**, deriving the penalty-from-prior equivalence the ML reader already half-knows.
- Reach for the **right distribution** from a catalog of fifteen-plus common priors — knowing each one's support, parameters, shape, and the situation that calls for it.
- Avoid the classic footguns: **precision-vs-σ** confusion inherited from BUGS, the **InverseGamma-near-zero pathology**, and treating priors on **location** and **scale** with the same logic when they need different logic.

---

## 1. What a prior is, and the myth of "no prior"

Let me start with the sentence that the entire chapter hangs on. A prior is a **probability distribution over a parameter, written before you condition on the current data, that quantifies what values of that parameter you consider plausible.** That's it. It is not a guess, not a fudge factor, not a confession of ignorance you're embarrassed about. It is a statement — in the honest language of probability — about which parameter values you would bet on and how heavily.

Recall the machinery from Chapter 01. Bayes' theorem multiplies a prior $p(\theta)$ by a likelihood $p(y \mid \theta)$ and renormalizes:

$$
p(\theta \mid y) = \frac{p(y \mid \theta)\, p(\theta)}{p(y)}, \qquad p(y) = \int p(y \mid \theta)\, p(\theta)\, d\theta.
$$

In ASCII, the shape that matters:

```
posterior  ∝  likelihood  ×  prior
p(θ|y)     ∝  p(y|θ)      ×  p(θ)
```

> 💡 **Intuition:** The likelihood is the data's voice; the prior is everything you knew before the data spoke. The posterior is the negotiation between them. If the data are loud (large $n$, strong signal), the likelihood dominates and the prior barely matters. If the data are quiet (small $n$, weak signal, or a parameter the data can't pin down — like a group-level variance with three groups), the prior does real work. A good Bayesian doesn't fear that; they *want* the prior to do real work exactly when the data can't, because the alternative is wild, unstable estimates.

### There is no "no prior"

Here is the claim people resist, so let me make it airtight. **You cannot avoid having a prior.** You can only choose between (a) a prior you wrote down on purpose, and (b) a prior that got chosen for you by a default, a software setting, or by pretending one isn't there.

Three angles on why:

**1. Maximum likelihood is itself a prior choice in disguise.** When a frequentist maximizes the likelihood, the result equals the *posterior mode* under a flat prior $p(\theta) \propto 1$. That flat "prior" is a genuine choice — and as we'll see in §2, it's frequently a *bad* one, because flat on $\theta$ is not flat on $\log\theta$ or $\theta^2$, so even "I'll just be neutral" silently privileges a particular parameterization.

**2. Regularization is a prior.** Every ML practitioner who has ever added an L2 penalty has used a Gaussian prior (we derive this in §5). Ridge, lasso, weight decay, early stopping, dropout-as-approximate-Bayes — these are all priors wearing engineering clothes. If you've trusted regularization to make your models generalize, you already trust priors; you just didn't call them that.

**3. The model structure is a prior.** The moment you write `mu = a + b*x` instead of a neural net, you've placed enormous prior mass on "the relationship is linear." Choosing a likelihood family — Normal vs Student-t vs Poisson — is a prior over how the data are generated. The prior over *parameters* is only the most visible instance of a commitment you're making everywhere.

> 📜 **Citation/Origin:** The "small world / large world" framing — your model is a small, self-consistent world, and the prior is part of its physics — is McElreath's (*Statistical Rethinking*, Ch 1–2). The point that there is no escape from priors, only the choice of which one, runs through all of Bayesian Data Analysis (Gelman et al., BDA3, Ch 2). The slogan I'd have you memorize: *the question is never whether to use a prior, only whether to use one you understand.*

### What "encoding" means concretely

When I say a prior *encodes* something, I mean you can read information out of it. Put $\theta \sim \mathrm{Normal}(0, 1)$ on a standardized regression slope and you've encoded: "before seeing data, I think a one-SD change in the predictor probably moves the outcome by somewhere between roughly $-2$ and $+2$ SD, most likely near zero, and a move of $+5$ SD would astonish me." That last clause is the load-bearing one. A prior earns its keep less by where it puts its mass than by where it *refuses* to — by ruling out the absurd. Hold that thought; it's the definition of "weakly informative" in §4.

---

## 2. Flat and improper priors — why flat ≠ uninformative

The most seductive idea in all of priorhood is: *"I don't want to bias anything, so I'll use a flat prior — equal weight everywhere — and let the data speak."* It sounds like the humble, objective choice. It is a trap, and learning exactly why is one of the most clarifying things you can do early in your Bayesian education.

### Improper priors: the integral that doesn't converge

A **proper** prior is an honest probability distribution: it integrates (or sums) to 1. A **flat** prior on an unbounded parameter — $p(\theta) \propto 1$ for $\theta \in \mathbb{R}$, or the scale-flat $p(\sigma) \propto 1/\sigma$ on $(0, \infty)$ — does *not* integrate to a finite number. We call such a thing an **improper prior**: it's a positive function, but not a probability distribution.

$$
\int_{-\infty}^{\infty} 1 \, d\theta = \infty \quad (\text{not a distribution}), \qquad \int_0^\infty \frac{1}{\sigma}\, d\sigma = \infty \quad (\text{also not}).
$$

Sometimes you can still get away with it: an improper prior times a likelihood can yield a *proper posterior* (one that does integrate to 1), and classical results lean on this. But "sometimes" is the problem. Here is the danger list, and every item has bitten real analysts:

**(a) The posterior can be improper too — silently.** In hierarchical models, a flat or near-flat prior on a group-level variance can produce a posterior that doesn't integrate. Your sampler won't print a tidy error; it'll just wander, produce divergences, or return nonsense that *looks* like a number. The eight-schools funnel (Chapter 07, Chapter 10) is the canonical place this bites. Separable logistic regression — where a predictor perfectly separates the classes — does the same thing: under a flat prior on the coefficients, the MLE is $\pm\infty$ and the posterior has no proper mode.

**(b) Flat is not invariant to reparameterization.** This is the deep one. Suppose $p(\theta) \propto 1$ is "uninformative" about $\theta$. Change variables to $\phi = \log\theta$. By the change-of-variables rule, the implied density on $\phi$ is

$$
p(\phi) = p(\theta)\left|\frac{d\theta}{d\phi}\right| = 1 \cdot e^\phi \propto e^\phi,
$$

which is **not** flat — it piles mass at large $\phi$. So "flat on $\theta$" and "flat on $\log\theta$" are different, contradictory states of belief. A prior that claims to encode "no information" but changes its mind when you take a logarithm was never encoding "no information" at all. It was encoding "I privilege this particular scale," which is information — usually accidental, usually wrong.

> 🧮 **The math:** This is why **Jeffreys priors** exist. Jeffreys' rule defines $p(\theta) \propto \sqrt{\det I(\theta)}$, where $I(\theta)$ is the Fisher information. It is constructed precisely to be *invariant under reparameterization*: do the change of variables and you get the Jeffreys prior for the new parameter automatically. For a binomial proportion it gives $\mathrm{Beta}(\tfrac12, \tfrac12)$; for a Normal mean (variance known) it gives the flat prior; for a Normal scale it gives $p(\sigma)\propto 1/\sigma$. Jeffreys priors are the principled face of "objective" / "noninformative" priors. But — and this matters for applied work — they are *frequently improper*, they get awkward and sometimes ill-defined in multi-parameter and hierarchical models, and they rarely express anything you actually believe. They're a beautiful idea you'll reach for far less than the textbooks imply.

**(c) Flat priors break the tools you'll come to rely on.** You cannot do a **prior predictive check** (Chapter 03) with an improper prior — you can't simulate from a distribution that isn't one. **Marginal likelihoods** and **Bayes factors** (Chapter 11) are undefined or arbitrary, because $p(y) = \int p(y\mid\theta)p(\theta)\,d\theta$ inherits the improper prior's infinite normalizer. A whole wing of the Bayesian house becomes uninhabitable.

> ⚠️ **Pitfall:** "Flat = uninformative" is false twice over. First, flat is **not** uninformative because it's not invariant — it secretly informs by privileging a scale. Second, flat is often **actively harmful**, because on an unbounded scale it puts *enormous* mass on absurd values. A flat prior on a standardized regression slope says a slope of $+10{,}000$ is exactly as plausible as a slope of $0$. That's not humility; that's a statement that you'd believe a one-SD nudge in $x$ could move $y$ by ten thousand SD. No human believes that. A weakly informative prior — which mildly says "probably between $-2$ and $+2$" — is **both** more honest *and* more numerically stable.

### Scale-dependence: the units trap

One more flavor of the flatness problem, because it shows up constantly in regression. Suppose you put a "vague" $\mathrm{Normal}(0, 1000)$ prior on a slope. Is that uninformative? It depends entirely on the **units** of your predictor. If $x$ is in meters, a slope of $1000$ might be wild; if $x$ is in kilometers, the *same physical relationship* has a slope $1000\times$ larger, and your "vague" prior is now tight. A prior is only "weak" or "strong" *relative to a scale*. This is the single best argument for **standardizing your predictors** (subtract the mean, divide by the SD) before setting priors — a practice we'll adopt as the default from Chapter 03 onward. On standardized predictors, a unit-scale prior like $\mathrm{Normal}(0,1)$ means the same thing regardless of whether the raw data were in meters or furlongs, because the standardized predictor is unitless by construction.

> 🔧 **In practice:** PyMC will *let* you use an improper-ish prior (e.g. `pm.Flat`, `pm.HalfFlat`, or a `Normal` with `sigma=1e6`) and it'll often sample fine on easy problems. Resist. The two-minute habit that prevents a category of bugs: standardize, then use proper, weakly-informative, unit-scale priors. Reserve `pm.Flat` for the rare, deliberate case where you've proven the posterior is proper and you have a specific reason (e.g., matching a classical result for pedagogy). The Stan team's official recommendation is blunt: a flat prior is *"not usually recommended."*

---

## 3. Proper priors and conjugacy — three updates you can do by hand

A **proper** prior integrates to 1 and is therefore a real distribution you can sample from, plot, and reason about. Almost everything you'll use in this course is proper. Within the proper priors there's a special, beautiful subset — the **conjugate** priors — where the math closes in your hands and the posterior comes out as a tidy formula. We're going to derive the three you must know cold, and *compute actual numbers*, because conjugacy is the cleanest possible lens on what Bayesian updating *does*. (In Chapter 04 we'll use these as exact checks on our samplers; here we use them to build intuition.)

> 💡 **Intuition:** A prior is **conjugate** to a likelihood when the posterior lands in the *same distributional family* as the prior. You start with a Beta, see binomial data, and end with a Beta. The data just *update the parameters*. When that happens, the parameters of the prior behave like **pseudo-counts** — pretend observations you're contributing before the real data arrive — and the posterior parameters are literally "pseudo-counts + real counts." Once you see updating as counting, conjugacy stops being algebra and becomes obvious.

### 3.1 Beta–Binomial: the proportion update

**Setup.** You're estimating a probability $\theta \in [0,1]$ — a conversion rate, a coin's bias, a free-throw percentage. The natural prior on a $[0,1]$ quantity is the **Beta**: $\theta \sim \mathrm{Beta}(\alpha, \beta)$, with density

$$
p(\theta) = \frac{\theta^{\alpha-1}(1-\theta)^{\beta-1}}{B(\alpha,\beta)} \;\propto\; \theta^{\alpha-1}(1-\theta)^{\beta-1}.
$$

You then observe $y$ successes in $n$ Bernoulli trials, so the likelihood is binomial: $p(y\mid\theta) \propto \theta^{y}(1-\theta)^{n-y}$.

**Derivation.** Multiply prior by likelihood and collect the exponents:

$$
p(\theta\mid y) \;\propto\; \underbrace{\theta^{\alpha-1}(1-\theta)^{\beta-1}}_{\text{prior}} \cdot \underbrace{\theta^{y}(1-\theta)^{n-y}}_{\text{likelihood}} \;=\; \theta^{(\alpha + y) - 1}(1-\theta)^{(\beta + n - y) - 1}.
$$

That's the kernel of a Beta distribution. So, with no integration at all:

$$
\boxed{\;\theta \mid y \;\sim\; \mathrm{Beta}\big(\alpha + y,\; \beta + n - y\big)\;}
$$

You **add successes to $\alpha$ and failures to $\beta$.** The posterior mean follows from the Beta mean $\mathbb{E}[\theta] = \alpha/(\alpha+\beta)$:

$$
\mathbb{E}[\theta\mid y] = \frac{\alpha + y}{\alpha + \beta + n}.
$$

> 🧮 **The math:** Rewrite that posterior mean as a weighted average to *see the shrinkage*. Let $w = \frac{\alpha+\beta}{\alpha+\beta+n}$. Then
> $$\mathbb{E}[\theta\mid y] = w\cdot\underbrace{\frac{\alpha}{\alpha+\beta}}_{\text{prior mean}} + (1-w)\cdot\underbrace{\frac{y}{n}}_{\text{MLE}}.$$
> The posterior mean is a **convex combination of the prior mean and the data's MLE.** The weight on the prior, $w$, depends on $\alpha+\beta$ — the prior's "sample size" in pseudo-counts. A $\mathrm{Beta}(2,2)$ contributes the equivalent of 4 pseudo-observations; a $\mathrm{Beta}(20,20)$ contributes 40 and pulls much harder. As real data $n \to \infty$, $w \to 0$ and the data win, exactly as they should.

**Numbers.** Say your prior is $\mathrm{Beta}(2, 2)$ — symmetric, gently favoring values near $\tfrac12$, prior mean $0.5$, worth 4 pseudo-counts. You run $n = 10$ trials and see $y = 7$ successes (MLE $= 0.70$).

$$
\theta\mid y \sim \mathrm{Beta}(2+7,\; 2+3) = \mathrm{Beta}(9, 5), \qquad \mathbb{E}[\theta\mid y] = \frac{9}{14} \approx 0.643.
$$

The posterior mean $0.643$ sits between the prior mean $0.5$ and the MLE $0.70$, pulled toward the prior because $n$ is small. The prior's weight is $w = 4/14 \approx 0.29$ — the prior is carrying about $29\%$ of the answer. Now imagine $n = 1000$ with $y = 700$: posterior $\mathrm{Beta}(702, 302)$, mean $\approx 0.699$, prior weight $w = 4/1004 \approx 0.004$. With a thousand trials the prior is a rounding error. **Same prior, opposite influence — and the difference is entirely the amount of data.** This is the whole story of priors in one example.

Two famous special cases worth memorizing:
- $\mathrm{Beta}(1, 1) = \mathrm{Uniform}(0, 1)$ — flat on the proportion (proper, because $[0,1]$ is bounded!). Its posterior mean is $\frac{1+y}{2+n}$, "Laplace's rule of succession."
- $\mathrm{Beta}(\tfrac12, \tfrac12)$ — the **Jeffreys prior** for the binomial; U-shaped, piling mass near 0 and 1.

We'll return to Beta–Binomial as the worked example of this chapter (§7), where we visualize the shrinkage by sweeping prior strength.

### 3.2 Gamma–Poisson: the rate update

**Setup.** You're estimating a rate $\lambda > 0$ — events per unit time, defects per wafer, goals per game. The conjugate prior is the **Gamma**: $\lambda \sim \mathrm{Gamma}(s, r)$ with **shape** $s$ and **rate** $r$ (this is the parameterization PyMC uses as `alpha=s, beta=r`), density $p(\lambda) \propto \lambda^{s-1} e^{-r\lambda}$. You observe counts $y_1, \dots, y_n$, each Poisson with mean $\lambda$, so $p(\mathbf{y}\mid\lambda) \propto \lambda^{\sum y_i} e^{-n\lambda}$.

**Derivation.**

$$
p(\lambda\mid\mathbf y) \propto \lambda^{s-1}e^{-r\lambda}\cdot \lambda^{\sum y_i}e^{-n\lambda} = \lambda^{(s + \sum y_i) - 1}\, e^{-(r + n)\lambda},
$$

which is the kernel of a Gamma. Hence:

$$
\boxed{\;\lambda \mid \mathbf y \;\sim\; \mathrm{Gamma}\Big(s + \textstyle\sum_i y_i,\; r + n\Big)\;}, \qquad \mathbb{E}[\lambda\mid\mathbf y] = \frac{s + \sum_i y_i}{r + n}.
$$

The interpretation is gorgeous: the **rate parameter $r$ acts like a number of prior "exposure" units, and the shape $s$ like a number of prior events.** You've pre-observed $s$ events in $r$ units of exposure; the data add $\sum y_i$ events over $n$ units; the posterior mean is total events over total exposure.

**Numbers.** Prior $\mathrm{Gamma}(s=3, r=2)$: prior mean $3/2 = 1.5$ events per unit, as if you'd seen 3 events in 2 units. You then observe $n = 5$ units with counts $\{2, 1, 3, 0, 2\}$, so $\sum y_i = 8$ (MLE $= 8/5 = 1.6$).

$$
\lambda\mid\mathbf y \sim \mathrm{Gamma}(3 + 8,\; 2 + 5) = \mathrm{Gamma}(11, 7), \qquad \mathbb{E}[\lambda\mid\mathbf y] = \frac{11}{7} \approx 1.571.
$$

Again the posterior mean ($1.571$) lands between prior mean ($1.5$) and MLE ($1.6$), nearer the data because the data's exposure ($n=5$) outweighs the prior's ($r=2$).

### 3.3 Normal–Normal: the mean update (the one that previews everything)

**Setup.** You're estimating an unknown mean $\mu$ from data with **known** measurement variance $\sigma^2$ (we relax "known" later in the course; assuming it here keeps the algebra clean and the lesson sharp). Prior $\mu \sim \mathrm{Normal}(\theta, \tau^2)$ — prior mean $\theta$, prior variance $\tau^2$. You observe $n$ points with sample mean $\bar y$.

**Result.** The posterior is Normal, $\mu\mid\mathbf y \sim \mathrm{Normal}(\mu_n, \tau_n^2)$, with

$$
\mu_n = \frac{\theta\,\sigma^2 + \bar y\, n\tau^2}{n\tau^2 + \sigma^2}, \qquad \tau_n^2 = \frac{\tau^2\sigma^2}{n\tau^2 + \sigma^2}.
$$

These look fiddly until you rewrite them in **precision** (precision = 1/variance), where they become unforgettable.

> 🧮 **The math:** Let prior precision be $\kappa_0 = 1/\tau^2$ and each datum's precision be $\kappa = 1/\sigma^2$. Then:
> $$\underbrace{\frac{1}{\tau_n^2}}_{\text{posterior precision}} = \underbrace{\frac{1}{\tau^2}}_{\text{prior precision}} + \underbrace{\frac{n}{\sigma^2}}_{\text{data precision}} \;=\; \kappa_0 + n\kappa,$$
> $$\mu_n = \frac{\kappa_0\,\theta + n\kappa\,\bar y}{\kappa_0 + n\kappa}.$$
> **Precisions add. The posterior mean is the precision-weighted average of the prior mean and the data mean.** Information, measured as precision, is additive — every observation contributes $\kappa = 1/\sigma^2$ worth of it, and the prior contributes $\kappa_0 = 1/\tau^2$ worth. This is the single most important formula to internalize in all of Bayesian inference, because it *is* the partial-pooling / shrinkage formula that powers hierarchical models in Chapter 10. There, the "prior mean" becomes the group-level mean estimated from the other groups, and shrinkage of each group toward the overall mean is exactly this precision-weighted average.

**Numbers.** Suppose you're estimating a true effect $\mu$. Prior $\mu \sim \mathrm{Normal}(\theta = 0,\; \tau^2 = 1)$ so $\kappa_0 = 1$. Your measurement noise has known $\sigma^2 = 4$ so $\kappa = 0.25$ per observation. You collect $n = 16$ observations with sample mean $\bar y = 3$.

- Data precision: $n\kappa = 16 \times 0.25 = 4$. Total precision: $\kappa_0 + n\kappa = 1 + 4 = 5$, so $\tau_n^2 = 1/5 = 0.2$.
- Posterior mean: $\mu_n = \dfrac{1\cdot 0 + 4\cdot 3}{5} = \dfrac{12}{5} = 2.4$.

The MLE was $\bar y = 3$; the prior mean was $0$; the posterior mean $2.4$ is pulled $20\%$ of the way back toward the prior, because the prior carries $1$ of the $5$ total precision units. Notice the posterior is also *tighter* than either input alone — its variance $0.2$ is smaller than both the prior variance $1$ and the per-observation variance $4$ — because combining independent sources of information always sharpens. That sharpening is the entire payoff of pooling.

> 🔧 **In practice:** You will rarely *fit* models by hand with these formulas — that's what the sampler is for. But you should reach for conjugate results constantly as **sanity checks and intuition pumps**: "if I doubled the data, the prior weight should roughly halve — does my posterior move accordingly?" When a fancy MCMC result contradicts the conjugate intuition on a simplified version of your model, trust the conjugate intuition and go find your bug. Chapter 04 makes this discipline explicit.

> 📜 **Citation/Origin:** Clean, verified update formulas with the pseudo-count framing live in *Bayes Rules!* Ch 5 (bayesrulesbook.com/chapter-5) and, exhaustively, in John D. Cook's *A Compendium of Conjugate Priors* (johndcook.com/CompendiumOfConjugatePriors.pdf) — the latter is the table to bookmark for the appendix cheat-sheet. BDA3 Ch 2–3 (Gelman et al.) is the reference derivation.

> ⚠️ **Pitfall — don't pick a prior because it's conjugate.** Conjugacy is a *convenience*, not a *reason*. In the BUGS/JAGS era people used InverseGamma priors on variances precisely because they're conjugate to the Normal — and it turns out that's one of the worst things you can do to a hierarchical variance (we'll see exactly why in §6 and at length in Chapter 03). Modern HMC/NUTS samplers (Chapter 05) don't need conjugacy at all; they'll happily integrate any proper prior you write. So choose the prior that *describes your beliefs and behaves well numerically*, and let the sampler handle the algebra. Conjugate families remain invaluable for teaching, for fast prototyping, and for exact checks — just not as a crutch for prior choice.

---

## 4. The spectrum: weakly informative ↔ informative ↔ regularizing

Now that we have proper priors, we can organize them along the axis that actually matters in practice: **how much do they constrain the answer?** These words get conflated constantly, so I'm going to pin each one down with a definition, a PyMC example, and the situation it's for. Picture a dial.

```
 noninformative        weakly informative        regularizing         informative
 (flat / Jeffreys)      Normal(0,1) on a            Laplace prior        Normal(0.4, 0.1)
 "let data speak"       standardized slope          (lasso) for          from a meta-analysis
   ⚠ fragile            rules out the absurd        sparsity             encodes real knowledge
        ◄───────────────────────────────────────────────────────────────────────►
              less constraint                              more constraint
```

**Noninformative / flat / "objective"** (Jeffreys, reference priors). The aspiration is to "let the data speak" and inject nothing. As §2 showed, this aspiration is half-illusory (flat isn't invariant) and operationally fragile (improper posteriors, broken predictive checks, instability in hierarchical models). It has a legitimate niche — matching classical results, certain reference analyses — but it is *not* your default, and the Stan team explicitly lists flat priors as "not generally recommended."

**Weakly informative.** This is your **default**, and it deserves the most attention. A weakly informative prior **rules out the absurd while staying agnostic within the plausible range.** It is deliberately wider than your actual beliefs — it's not trying to be right, just trying to keep the model sane. On a standardized predictor, $\beta \sim \mathrm{Normal}(0, 1)$ says "a slope beyond about $\pm 3$ would be very surprising," which on unitless standardized data is true for essentially every real problem, while imposing almost nothing within $[-3, 3]$. The payoff is twofold: (1) it stabilizes inference when data are scarce or partially separated, and (2) it makes your model *generative* — you can simulate fake data from it and check it (Chapter 03). 

> 📜 **Citation/Origin — the Stan Prior-Choice hierarchy.** The Stan team maintains a 5-level ladder of priors (github.com/stan-dev/stan/wiki/prior-choice-recommendations), which I'd have you memorize because it's the most concrete advice anywhere:
> 1. **Flat** — *not recommended.*
> 2. **Super-vague but proper**, e.g. `normal(0, 1e6)` — *not recommended* (all the dangers of flat, dressed as proper).
> 3. **Very weakly informative**, e.g. `normal(0, 10)` — okay, often still too wide on standardized scales.
> 4. **Generic weakly informative**, e.g. **`normal(0, 1)`** (Gelman) or **`student_t(3, 0, 1)`** (Vehtari — heavier tails for robustness) — *the recommended default.*
> 5. **Specific informative** — when you genuinely have domain knowledge.
>
> For regression coefficients with predictors scaled to mean 0 and SD 0.5, current Stan-wiki guidance suggests `student_t(ν, 0, s)` with $3 < ν < 7$, and rstanarm's default is `normal(0, 2.5)`. The historic `cauchy(0, 2.5)` ($=$ `student_t(1, 0, 2.5)`) default is now *discouraged* — its tails are so heavy they cause sampling trouble when the data are only weakly informative.

**Informative.** A genuinely informative prior **encodes real domain knowledge** and is narrow enough to influence the posterior even with moderate data. $\beta \sim \mathrm{Normal}(0.4, 0.1)$ — "the effect is around 0.4, I'd be surprised outside $[0.2, 0.6]$" — might come from a meta-analysis, a previous experiment, or hard physical constraints. Informative priors are *good and underused*; the taboo against them is mostly cultural. The discipline they demand: be ready to defend where the information came from, and run a **sensitivity analysis** (Chapter 03) to show how much the conclusion leans on the prior versus the data.

**Regularizing.** A regularizing prior **deliberately shrinks parameters toward a null** (usually zero) to trade a little bias for a lot of variance reduction — exactly the bias–variance bargain you know from ML. A Gaussian prior gives ridge-style shrinkage; a Laplace prior gives lasso-style sparsity; the **horseshoe** gives aggressive, adaptive sparsity (most coefficients crushed to zero, a few allowed to escape). The next section makes the ridge/lasso connection exact, and the horseshoe gets its full treatment in Chapter 03 (concept) and Chapter 14 (sparse models).

> 💡 **Intuition:** "Weakly informative" and "regularizing" overlap heavily — a $\mathrm{Normal}(0,1)$ on a slope both *rules out the absurd* (weakly informative framing) and *shrinks toward zero* (regularizing framing). The difference is intent and emphasis. You call it **weakly informative** when your goal is keeping the model sane and honest; you call it **regularizing** when your goal is explicit shrinkage for predictive performance or variable selection. Same object, different job description. Don't let the vocabulary fool you into thinking they're different mechanisms.

---

## 5. Ridge and lasso are Gaussian and Laplace priors (the bridge from ML)

If you've done machine learning, you already use priors — you call them penalties. Let me make the equivalence exact, because once you see it, regularization and Bayesian priors stop being two topics and become one. The key fact: **the MAP (maximum a posteriori) estimate under a prior equals the penalized-likelihood estimate where the penalty is the negative log-prior.**

### The general equivalence

The MAP estimate maximizes the posterior, equivalently it maximizes the *log* posterior:

$$
\hat\theta_{\text{MAP}} = \arg\max_\theta \big[\log p(y\mid\theta) + \log p(\theta)\big] = \arg\min_\theta \big[\underbrace{-\log p(y\mid\theta)}_{\text{loss}} \;\underbrace{-\log p(\theta)}_{\text{penalty}}\big].
$$

So **every prior induces a penalty equal to its negative log-density, and every penalty corresponds to a prior** (whatever distribution has that penalty as its negative log-density, after normalizing). Regularized ML is Bayesian MAP estimation. Let's specialize.

### Ridge = Gaussian prior

Take a linear-Gaussian regression $y_i \sim \mathrm{Normal}(\mathbf{x}_i^\top\boldsymbol\beta,\; \sigma^2)$, so the negative log-likelihood is $\frac{1}{2\sigma^2}\sum_i (y_i - \mathbf x_i^\top\boldsymbol\beta)^2$ plus a constant. Put an i.i.d. **Gaussian** prior on each coefficient: $\beta_j \sim \mathrm{Normal}(0, \tau^2)$, whose negative log-density is $\frac{1}{2\tau^2}\beta_j^2$ plus a constant. The MAP objective is

$$
\min_{\boldsymbol\beta}\; \frac{1}{2\sigma^2}\sum_i (y_i - \mathbf x_i^\top\boldsymbol\beta)^2 + \frac{1}{2\tau^2}\sum_j \beta_j^2 \;\;\equiv\;\; \min_{\boldsymbol\beta}\; \sum_i (y_i - \mathbf x_i^\top\boldsymbol\beta)^2 + \lambda \sum_j \beta_j^2,
$$

after multiplying through by $2\sigma^2$ and defining $\boxed{\lambda = \sigma^2/\tau^2}$. **That is exactly ridge regression / L2 regularization / weight decay.** The ridge penalty strength $\lambda$ is the ratio of noise variance to prior variance: a *tighter* prior (smaller $\tau^2$) means *stronger* regularization (larger $\lambda$), which is precisely what your intuition should say. Ridge has always been MAP estimation under a Gaussian prior; nobody told you because the two communities used different words.

### Lasso = Laplace prior

Now swap the Gaussian prior for a **Laplace** (double-exponential) prior: $\beta_j \sim \mathrm{Laplace}(0, b)$, density $\frac{1}{2b}\exp(-|\beta_j|/b)$, whose negative log-density is $\frac{1}{b}|\beta_j|$ plus a constant. The MAP objective becomes

$$
\min_{\boldsymbol\beta}\; \frac{1}{2\sigma^2}\sum_i (y_i - \mathbf x_i^\top\boldsymbol\beta)^2 + \frac{1}{b}\sum_j |\beta_j| \;\;\equiv\;\; \min_{\boldsymbol\beta}\; \sum_i (y_i - \mathbf x_i^\top\boldsymbol\beta)^2 + \lambda \sum_j |\beta_j|,
$$

with $\lambda = 2\sigma^2/b$. **That is exactly the lasso / L1 regularization.** The Laplace prior's defining feature is its **sharp peak at zero** (the $|\beta|$ kink) — that non-differentiable spike is what drives coefficients *exactly* to zero in the MAP solution, producing sparsity, whereas the smooth Gaussian only shrinks them *toward* zero without ever zeroing them out.

> ⚠️ **Pitfall — the Bayesian lasso doesn't give you a sparse posterior.** Here's a subtlety that trips up people who learn the equivalence and over-extend it. The *MAP* under a Laplace prior is sparse (exact zeros), but the *full posterior* under a Laplace prior is **not** — the posterior mean of each coefficient is essentially never exactly zero, and the Laplace tails aren't actually heavy enough to leave large true coefficients unshrunk. So "lasso = Laplace prior" is true *for the point estimate*, but if you want genuine Bayesian variable selection — strong shrinkage of noise coefficients while letting real signals through unscathed — you want the **horseshoe** or **regularized horseshoe**, not the Laplace. We flag it here and develop it in Chapter 03 and Chapter 14. The lesson: the prior↔penalty bridge is exact at the MAP, but a full posterior is richer than its mode.

> 💡 **Intuition:** Why does L2 shrink-but-keep while L1 zeros-out? Look at the priors' shapes. The Gaussian is *round* at zero — smooth, no kink — so the penalty's gradient vanishes as $\beta\to 0$, and there's no force to push a small coefficient the last little bit to exactly zero. The Laplace has a *corner* at zero — a constant-magnitude gradient $\pm 1/b$ right up to the origin — so there's always a finite force pushing toward zero, and once the data's pull is weaker than that force, the coefficient snaps to exactly zero and stays. Sparsity is a geometric consequence of the prior's kink. You can *see* the whole story in the two density plots, which is why I'll have you draw them in the exercises.

> 🔧 **In practice (PyMC):** The Bayesian versions are one line each. Ridge-style: `beta = pm.Normal("beta", 0, tau, dims="predictor")`. Lasso-style: `beta = pm.Laplace("beta", mu=0, b=b, dims="predictor")`. But — and this is the point of *going* Bayesian — you usually shouldn't fix the shrinkage strength by hand. Put a prior on $\tau$ (or $b$) and let the model learn how much to regularize from the data. That's a tiny hierarchical model, and it's strictly better than cross-validating a single $\lambda$ because it propagates the uncertainty about the regularization strength into your final intervals. This "learn the regularization" move is one of the quiet superpowers of the Bayesian approach.

---

## 6. The catalog: common distributions and exactly when to reach for each

This is the reference you'll return to. Below is the field guide to the priors you'll actually use, organized by what they're *for*. Before the table, the one warning that prevents more bugs than any other.

> ⚠️ **THE precision footgun: PyMC and Stan use the standard deviation, not the variance, and DEFINITELY not the precision.** When you write `pm.Normal("mu", mu=0, sigma=1)`, that `1` is the **standard deviation $\sigma$** — not the variance $\sigma^2$, and not the precision $\tau = 1/\sigma^2$. This matters enormously because the previous generation of Bayesian software — **BUGS and JAGS** — parameterized the Normal by its **precision**. Decades of textbooks, StackExchange answers, and copied-and-pasted model snippets are written in precision. If you transcribe a BUGS model `mu ~ dnorm(0, 0.001)` literally into PyMC as `pm.Normal("mu", 0, 0.001)`, you have just put a prior with **SD = 0.001** — absurdly tight, locking `mu` to essentially zero — when the BUGS author meant **precision = 0.001**, i.e. **SD $= 1/\sqrt{0.001} \approx 31.6$**, an extremely *wide* prior. That's a factor of 30,000 error in the wrong direction, and it will silently wreck your inference. **Every time you read a Normal prior from older Bayesian material, ask: is that number an SD or a precision?** In PyMC it is always an SD.

### How to read the catalog

For each distribution I give: its **support** (what values it can take), its **PyMC spelling** (with the exact parameter names), its **shape/behavior**, and **when to reach for it**. I've split the catalog into priors for **unbounded** quantities (locations, coefficients), priors for **positive** quantities (scales, rates), priors for **bounded** quantities (probabilities, proportions), priors for **simplex/correlation** structures, and a couple of specialists.

#### 6.1 Priors for unbounded quantities (locations, intercepts, slopes)

| Distribution | Support | PyMC spelling | Shape & behavior | When to reach for it |
|---|---|---|---|---|
| **Normal** | $\mathbb{R}$ | `pm.Normal("b", mu=0, sigma=1)` | Symmetric bell, light (Gaussian) tails. | The **default** for locations, intercepts, and coefficients on standardized predictors. `Normal(0,1)` is the generic weakly-informative workhorse. |
| **StudentT** | $\mathbb{R}$ | `pm.StudentT("b", nu=3, mu=0, sigma=1)` | Bell with **heavy tails** controlled by $\nu$; $\nu\to\infty$ recovers Normal; small $\nu$ allows occasional large values. | A **robust** alternative to Normal — for coefficients you want regularized but not strangled, and as a robust likelihood for outlier-prone data. Vehtari's recommended weakly-informative default is `student_t(3, 0, 1)`. |
| **Cauchy** | $\mathbb{R}$ | `pm.Cauchy("b", alpha=0, beta=1)` | StudentT with $\nu = 1$: tails so heavy the **mean and variance don't exist**. | Rarely now. Historically a common weakly-informative default for logistic-regression coefficients (Gelman, Jakulin, Pittau & Su, 2008, *Ann. Appl. Stat.* 2(4):1360–1383), but **discouraged today** (current Stan-wiki guidance) because its tails are heavy enough to slow or stall NUTS. Reach for it only when you truly want to entertain wild values; otherwise use StudentT with $\nu \ge 3$. |
| **Laplace** | $\mathbb{R}$ | `pm.Laplace("b", mu=0, b=1)` | Sharp **peak (kink) at the location**, exponential tails. | **Lasso-style** sparsity in MAP estimation (§5); occasionally as a mildly robust location prior. Remember the §5 caveat: it does *not* give a sparse full posterior. |

> 💡 **Intuition — light vs heavy tails.** The whole game in choosing among Normal / StudentT / Cauchy / Laplace for a location is **how much you want to penalize large values.** Normal penalizes them quadratically — a coefficient of 5 costs $25\times$ more log-prior than a coefficient of 1 — so it pulls hard toward zero and *strongly resists* outliers. Heavy-tailed StudentT and Cauchy penalize large values much more gently, so they let a genuinely large coefficient or a genuine outlier "escape" the shrinkage instead of being dragged in. Laplace sits in between on tails but adds a peak that creates sparsity. Pick the tail weight that matches your belief about how surprising large values would be.

#### 6.2 Priors for positive quantities (scales, standard deviations, rates)

These go on things that **must be positive** — a standard deviation $\sigma$, a rate $\lambda$, a lengthscale, a dispersion. The logic here is genuinely *different* from location priors (see §6.6), and conflating them is a top source of pathologies.

| Distribution | Support | PyMC spelling | Shape & behavior | When to reach for it |
|---|---|---|---|---|
| **HalfNormal** | $(0,\infty)$ | `pm.HalfNormal("sigma", sigma=1)` | Right half of a Normal; mode at 0, **light tail**. Mean $=\sigma\sqrt{2/\pi}\approx 0.8\sigma$. | **Excellent default for scale parameters** on standardized data. Light tail keeps things tame; mild push toward small scales. |
| **Exponential** | $(0,\infty)$ | `pm.Exponential("sigma", lam=1)` | Monotone decreasing from 0; **single rate parameter `lam`**; mean $= 1/\lambda$. | **Clean one-parameter default** for scales (PyMC docs favor it). `Exponential(1)` has mean 1 — sensible on unit scale. Light-ish tail. |
| **HalfCauchy** | $(0,\infty)$ | `pm.HalfCauchy("sigma", beta=1)` | Mode at 0, **very heavy tail** — permits large scales without ruling them out. | Gelman's (2006) recommended **hierarchical-variance** prior when you genuinely might have a large group-level SD (e.g. eight-schools $\tau$, scale 25). Use when you don't want to foreclose large scales — but watch for slow NUTS. |
| **half-StudentT** | $(0,\infty)$ | `pm.HalfStudentT("sigma", nu=4, sigma=1)` | Half of a StudentT; tunable tail between HalfNormal ($\nu\to\infty$) and HalfCauchy ($\nu=1$). | The **tunable compromise**: `half-t(4, 0, 1)` is the Stan-wiki refinement when groups are few — heavier tail than HalfNormal, tamer than HalfCauchy. |
| **Gamma** | $(0,\infty)$ | `pm.Gamma("x", alpha=2, beta=0.1)` *or* `pm.Gamma("x", mu=..., sigma=...)` | Flexible: `alpha<1` spikes at 0, `alpha=1` is Exponential, `alpha>1` bumps away from 0. **`beta` is the rate.** | Conjugate prior for Poisson rates (§3.2); a flexible positive prior; `Gamma(2, ...)` is a nice **boundary-avoiding** shape that's zero *at* the origin, handy for parameters you don't want pinned at 0. |
| **InverseGamma** | $(0,\infty)$ | `pm.InverseGamma("v", alpha=3, beta=0.5)` | Heavy right tail; **pathological near 0** for small parameters (see warning). | Conjugate prior for a Normal *variance* — and a **trap**. Use only with deliberate, non-tiny parameters (e.g. as a *slab width* in the regularized horseshoe). **Do not** use `InvGamma(ε, ε)` on a hierarchical variance. |
| **LogNormal** | $(0,\infty)$ | `pm.LogNormal("x", mu=0, sigma=1)` | $\exp$ of a Normal; **right-skewed**, multiplicative. `mu`/`sigma` are on the **log scale**. | Positive quantities that vary **multiplicatively / over orders of magnitude** — concentrations, incomes, GP lengthscales, anything where "factor of 2" is the natural unit. |

> ⚠️ **The InverseGamma-near-zero pathology — burn this in.** The conjugate prior for a Normal variance $\sigma^2$ is InverseGamma, and for *decades* people used `InverseGamma(ε, ε)` with tiny $\varepsilon$ (like 0.001) as a "noninformative" variance prior, because it's conjugate and looks vague. **Gelman (2006) showed it is genuinely pathological for hierarchical variance parameters.** Two failures: (i) as $\varepsilon \to 0$ there is *no proper limiting posterior* — so your "vague" prior has no well-defined vague limit; and (ii) for any small $\varepsilon$, the prior has a sharp spike that artificially **pulls the group-level SD toward 0** (spurious shrinkage to "no group variation"), and the inference is *highly sensitive to the exact $\varepsilon$* — especially when the number of groups is small or the true SD is near zero, which is *exactly* the regime where you can least afford a brittle prior. The cure is simple and it's the modern default: **put a HalfNormal, Exponential, or half-t prior on the SD $\sigma$ directly — never `InvGamma(ε, ε)` on the variance $\sigma^2$.** This is the single most important "don't" in the whole prior catalog, and Chapter 03 devotes serious time to it.

> 📜 **Citation/Origin:** Gelman, A. (2006), *"Prior distributions for variance parameters in hierarchical models,"* Bayesian Analysis 1(3):515–534 (sites.stat.columbia.edu/gelman/research/published/taumain.pdf). The half-Cauchy-on-$\sigma$ recommendation and the InverseGamma critique both come from this paper; it's required reading and we cite it again in Chapter 03 and Chapter 10.

#### 6.3 Priors for bounded quantities (probabilities, proportions, fractions)

| Distribution | Support | PyMC spelling | Shape & behavior | When to reach for it |
|---|---|---|---|---|
| **Beta** | $(0,1)$ | `pm.Beta("p", alpha=2, beta=2)` | Enormously flexible on $[0,1]$: `(1,1)`=Uniform, `(0.5,0.5)`=U-shaped Jeffreys, `(>1,>1)`=unimodal bump, asymmetric when $\alpha\neq\beta$. | The **default for a single probability or proportion** — conversion rates, biases, fractions. Conjugate to the Binomial (§3.1); $\alpha,\beta$ are pseudo-counts. |
| **Uniform** | $[a,b]$ | `pm.Uniform("x", lower=0, upper=1)` | Flat on a **bounded** interval — proper because bounded. | When a parameter has genuine **hard bounds** and you truly have no preference inside them. Use sparingly: hard edges can cause sampler trouble, and "flat inside" is rarely your real belief. Prefer Beta on $[0,1]$. |
| **Kumaraswamy** | $(0,1)$ | `pm.Kumaraswamy("p", a=2, b=2)` | Beta-*like* shapes on $[0,1]$ but with a **closed-form CDF and quantile function**. | A lightweight, computationally convenient Beta substitute — handy when you need cheap inverse-CDF sampling or a closed-form CDF. In most modeling you'll just use Beta; reach for Kumaraswamy when its analytic tractability buys you something specific. |

#### 6.4 Priors for simplexes and correlation structures

| Distribution | Support | PyMC spelling | Shape & behavior | When to reach for it |
|---|---|---|---|---|
| **Dirichlet** | simplex (vector summing to 1, all $\ge 0$) | `pm.Dirichlet("w", a=np.ones(k))` | Multivariate Beta over a $k$-simplex. `a=ones` is uniform on the simplex; larger entries concentrate near the center; entries $<1$ push mass toward the corners (sparsity). | **Mixture weights** (Chapter 13), category probabilities, any set of proportions that must sum to 1. The conjugate prior for the Multinomial. |
| **LKJ (correlation)** | correlation matrices | `pm.LKJCholeskyCov(...)` / `pm.LKJCorr(...)` | Single shape $\eta$. Density $\propto \det(R)^{\eta-1}$. **$\eta=1$ uniform over correlation matrices; $\eta>1$ concentrates near the identity (low correlation); $\eta<1$ favors strong correlations.** | The standard prior on **correlation matrices** for correlated varying effects (varying intercepts *and* slopes, Chapter 10) and multivariate outcomes. Use the **Cholesky** form `LKJCholeskyCov` for efficient, well-conditioned sampling. |

> ⚠️ **The LKJ direction everyone gets backwards.** With $\mathrm{LKJ}(\eta)$, **larger $\eta$ means *less* correlation**, not more. $\eta = 1$ is uniform; crank $\eta$ up and you concentrate prior mass on the *identity* matrix (variables independent). The mild-regularization default for varying-slope models is $\eta \approx 2$ — the correlation-matrix analogue of a gentle $\mathrm{Beta}(2,2)$, nudging toward independence without forbidding correlation. If your fitted correlations come out implausibly near zero, suspect $\eta$ is too large. (Full treatment in Chapter 10.) The PyMC idiom has its own footgun, the `sd_dist` argument, which must be built with the `.dist()` classmethod:
> ```python
> with pm.Model() as m:
>     sd_dist = pm.Exponential.dist(1.0, size=n)      # note .dist() — an *unregistered* dist, NOT a named RV
>     chol, corr, sigmas = pm.LKJCholeskyCov(
>         "chol_cov", eta=2, n=n, sd_dist=sd_dist, compute_corr=True
>     )
>     vals = pm.MvNormal("vals", mu=np.zeros(n), chol=chol)
> ```
> Passing a *named* RV (or the wrong shape) to `sd_dist` is the classic error; `compute_corr=True` returns the `(chol, corr, sigmas)` tuple. We use this exact pattern in Chapter 10.

#### 6.5 The full quick-reference card

For when you just need the answer fast:

```
PARAMETER TYPE                  REACH FOR                              PyMC
─────────────────────────────────────────────────────────────────────────────────────
intercept / location            Normal(0, s)                           pm.Normal
slope on standardized x          Normal(0, 1)  or  StudentT(3, 0, 1)    pm.Normal / pm.StudentT
coefficient, want robustness     StudentT(3..7, 0, s)                   pm.StudentT
coefficient, want sparsity (MAP) Laplace(0, b)                          pm.Laplace
scale / SD (default)             HalfNormal(s)  or  Exponential(1)      pm.HalfNormal / pm.Exponential
scale / SD (few groups)          half-t(4, 0, 1)                        pm.HalfStudentT
scale / SD (might be large)      HalfCauchy(s)                          pm.HalfCauchy
Poisson rate (conjugate)         Gamma(shape, rate)                     pm.Gamma
positive, multiplicative         LogNormal(0, s)                        pm.LogNormal
probability / proportion         Beta(a, b)                             pm.Beta
hard-bounded, no preference      Uniform(a, b)                          pm.Uniform
mixture / category weights       Dirichlet(ones(k))                     pm.Dirichlet
correlation matrix               LKJ(eta≈2), Cholesky form              pm.LKJCholeskyCov
────────────────────────  NEVER: InvGamma(ε, ε) on a hierarchical variance  ──────────
```

#### 6.6 Priors on location vs scale — why the logic is different

I keep insisting that location priors and scale priors obey different logic, so let me make the difference explicit, because getting it wrong is a recurring source of pathology.

A **location** parameter (an intercept, a slope, a mean) lives on the whole real line and is *symmetric* in a meaningful sense — moving it $+1$ or $-1$ are mirror-image operations. Symmetric, real-line priors (Normal, StudentT) are the natural fit. The question you ask is "how far from zero might this plausibly be?", and the answer sets the SD. The **additive** structure of the problem ($\mu = a + bx$) matches the additive symmetry of the prior.

A **scale** parameter (a standard deviation, a rate, a lengthscale, a dispersion) is fundamentally different in three ways:

1. **It's bounded below by zero** — there's a hard wall at 0, so you need a positive-support distribution (HalfNormal, Exponential, half-t, Gamma, LogNormal), never a Normal.
2. **It's multiplicative, not additive.** Doubling a scale and halving it are the "mirror image" operations, not adding and subtracting. That's why a LogNormal — symmetric *on the log scale* — is often the most natural scale prior, and why people reason about scales in orders of magnitude ("is it around 0.1, 1, or 10?").
3. **The behavior near zero is delicate and consequential.** A location prior near its center is unremarkable. But a scale prior's behavior *at zero* determines whether your group-level variation can collapse to "no variation at all." A prior that spikes at 0 (like `InvGamma(ε,ε)`, §6.2) forces spurious collapse; a prior that's *zero at the origin* (like `Gamma(2, ·)`) actively keeps you off the boundary. This boundary behavior has no analogue for location priors and is exactly why Gelman (2006) is a paper about *variance* priors specifically.

> 💡 **Intuition:** Set a **location** prior by asking *"how far from the center, additively, is plausible?"* Set a **scale** prior by asking *"what's the order of magnitude, and could it be essentially zero?"* Different questions, different distributions, different failure modes. When you find yourself about to put a `Normal` on something that can't be negative, stop — that's the tell that you're applying location logic to a scale.

> 🔧 **In practice:** On standardized data (outcome and predictors scaled to roughly unit SD, the default from Chapter 03 on), residual and group-level SDs are themselves order-1 quantities, so `pm.HalfNormal("sigma", 1)` or `pm.Exponential("sigma", 1)` are sensible weakly-informative scale priors out of the box. This is *another* reason standardization is the default: it puts both your location priors *and* your scale priors onto a common, interpretable unit scale, so a single mental template ("unit-scale weakly-informative") covers your whole model.

---

## 7. A principled lens: PC priors (penalized complexity)

Before the worked example, one more idea worth meeting now and developing in Chapter 03, because it gives a *principle* behind the "shrink toward simple, penalize complexity" instinct we've been leaning on all chapter. **Penalized-complexity (PC) priors** (Simpson, Rue, Riebler, Martins & Sørbye, 2017) are built from four principles:

1. **Occam's razor.** Prefer the *simpler* model component; make the prior favor it unless the data say otherwise.
2. **A complexity measure.** Measure how far a model component sits from a "base model" (e.g. a group-level SD of 0 = "no group effect"; a spline penalty of 0 = "straight line") using the **Kullback–Leibler divergence**, then transform it to a natural distance $d = \sqrt{2\,\mathrm{KL}}$.
3. **Constant-rate penalization.** Penalize that distance at a constant rate — which, mathematically, makes the prior an **Exponential distribution on the distance scale**, $p(d) = \mu e^{-\mu d}$.
4. **User-interpretable scaling.** Set the single rate via a meaningful probability statement like $P(\sigma > U) = a$ ("I'm 99% sure the group SD is below $U$").

> 💡 **Intuition:** PC priors formalize the gut feeling that *"flexibility should have to earn its keep."* You define the boring base model (zero effect, no curvature, independence), measure complexity as distance from it, and place an Exponential penalty on that distance. The beautiful payoff: an Exponential on the KL-distance scale often pulls back to a familiar prior on the natural parameter — for a Gaussian scale it lands you near an **Exponential / half-Normal-flavored prior on $\sigma$**, which is exactly the recommendation we reached pragmatically in §6.2. PC priors are a *derivation* of the practices the Stan wiki and Gelman (2006) arrived at empirically.

> 📜 **Citation/Origin:** Simpson, Rue, Riebler, Martins, Sørbye (2017), *"Penalising model component complexity: A principled, practical approach to constructing priors,"* Statistical Science 32(1):1–28 (arXiv:1403.4630). The exact closed form is model-specific — derive it from §2–3 of the paper for a given parameter (GP range, AR coefficient) rather than guessing. We use PC priors again when we discuss GP lengthscale priors in Chapter 12.

---

## 8. Worked example: Beta–Binomial conjugacy and a prior-strength sweep

Time to make the abstractions concrete and *see* shrinkage with our own eyes. We'll take the cleanest possible setting — a single proportion, the Beta–Binomial of §3.1 — and do two things: confirm that PyMC's sampler reproduces the analytic conjugate posterior (a habit from Chapter 04 you'll thank yourself for), and then **sweep the prior strength** to visualize how the posterior slides from "all data" to "all prior." This is the canonical demonstration that *the same data, under different priors, gives different — and predictably different — answers.*

**The scenario.** You're estimating a click-through rate $\theta$ for a new button. You run a small test: $n = 20$ impressions, $y = 13$ clicks. The MLE is $\hat\theta = 13/20 = 0.65$ — but with only 20 trials, that estimate is shaky, and you have a prior belief (from every button you've ever shipped) that CTRs cluster around $0.5$. How much should you trust the data versus your experience? The prior strength dial answers exactly that.

### 8.1 Setup and the analytic posterior

```python
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import scipy.stats as stats

RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)

# Observed data: 13 clicks in 20 impressions
y, n = 13, 20
mle = y / n                      # 0.65

# We'll sweep the prior STRENGTH while holding the prior MEAN fixed at 0.5.
# A symmetric Beta(a, a) has mean 0.5 and "pseudo-count" strength = 2a.
# Small a  -> weak prior (data dominates); large a -> strong prior (prior dominates).
prior_strengths = {
    "very weak  Beta(1,1)=Uniform": 1.0,
    "weak       Beta(2,2)":         2.0,
    "moderate   Beta(8,8)":         8.0,
    "strong     Beta(40,40)":      40.0,
}

# Analytic posterior for each: Beta(a + y, a + n - y)   [Section 3.1]
print(f"MLE = {mle:.3f}   (y={y}, n={n})\n")
for label, a in prior_strengths.items():
    post_a, post_b = a + y, a + (n - y)
    post_mean = post_a / (post_a + post_b)
    w = (2 * a) / (2 * a + n)            # weight on the prior (Section 3.1)
    # A genuine 94% HDI (ArviZ default), not an equal-tailed ppf interval:
    draws = stats.beta.rvs(post_a, post_b, size=200_000, random_state=0)
    lo, hi = az.hdi(draws, hdi_prob=0.94)
    print(f"{label:30s} posterior Beta({post_a:.0f},{post_b:.0f})  "
          f"mean={post_mean:.3f}  prior_weight={w:.2f}  94% HDI≈[{lo:.2f},{hi:.2f}]")
```

Running this prints a table you should read carefully — it's the whole lesson in four lines:

```
MLE = 0.650   (y=13, n=20)

very weak  Beta(1,1)=Uniform   posterior Beta(14,8)   mean=0.636  prior_weight=0.09  94% HDI≈[0.45,0.82]
weak       Beta(2,2)           posterior Beta(15,9)   mean=0.625  prior_weight=0.17  94% HDI≈[0.44,0.80]
moderate   Beta(8,8)           posterior Beta(21,15)  mean=0.583  prior_weight=0.44  94% HDI≈[0.43,0.73]
strong     Beta(40,40)         posterior Beta(53,47)  mean=0.530  prior_weight=0.80  94% HDI≈[0.44,0.62]
```

> 🩺 **How to read this table.** Scan the `mean` column top to bottom: it slides from **0.636** (almost the MLE of 0.65) down to **0.530** (almost the prior mean of 0.5). The `prior_weight` column tells you *why* — it's the $w$ from §3.1, the fraction of the answer the prior is carrying, rising from 9% to 80% as the prior's pseudo-count strength ($2a$) grows from 2 to 80 against the data's $n=20$ (the strongest prior, with pseudo-count $80$, swamps the $n=20$ data: $80/100 = 0.80$). And look at the HDI width: the strong prior doesn't just shift the estimate, it *narrows* it (from $\approx 0.37$ wide to $\approx 0.19$ wide), because a strong prior contributes a lot of precision. That narrowing is a warning as much as a feature — an over-confident prior gives you a tight interval *in the wrong place*. The cure is never "trust the tight interval"; it's a prior predictive check (Chapter 03) to confirm the prior was reasonable before you ever looked at this table.

### 8.2 Confirming with PyMC's sampler

You wouldn't normally MCMC a problem with a closed form — but doing it *once* here builds the habit of validating a sampler against known truth (Chapter 04, Chapter 08's fake-data recovery). We fit the moderate $\mathrm{Beta}(8,8)$ case.

```python
a = 8.0
with pm.Model() as beta_binom:
    theta = pm.Beta("theta", alpha=a, beta=a)               # prior
    pm.Binomial("y_obs", n=n, p=theta, observed=y)          # likelihood
    idata = pm.sample(
        2000, tune=1000, chains=4, target_accept=0.9,
        random_seed=RANDOM_SEED,                            # reproducible
    )                                                       # returns InferenceData

print(az.summary(idata, var_names=["theta"]))   # default kind="all": stats + r_hat/ess diagnostics
# Analytic check:
post_a, post_b = a + y, a + (n - y)                          # Beta(21, 15)
print("analytic posterior mean:", post_a / (post_a + post_b))   # 0.583
```

> 🩺 **Diagnostic — what to look at and what you should see.** `az.summary` reports `mean`, `sd`, the 94% HDI, plus `r_hat`, `ess_bulk`, and `ess_tail`. You want **`r_hat` ≤ 1.01** (chains agree — our threshold from Chapter 07), **`ess_bulk` and `ess_tail` ≳ 400** (enough effective samples for stable estimates), and **0 divergences** (PyMC prints a warning if any occur — there won't be any on a problem this smooth). The `mean` should print as ≈ **0.583**, matching the analytic value to within Monte Carlo error — confirming the sampler is telling the truth. Add `az.plot_trace(idata, var_names=["theta"])` to *see* it: you'll get a smooth, unimodal posterior density on the left and four "fuzzy caterpillar" chains fully overlapping on the right, the picture of healthy mixing. (We dwell on every one of these diagnostics in Chapter 07; here just confirm they look clean.)

### 8.3 Visualizing the sweep

The plot that makes shrinkage unforgettable overlays each posterior density against the fixed data likelihood and each prior:

```python
grid = np.linspace(0, 1, 500)
fig, axes = plt.subplots(1, len(prior_strengths), figsize=(16, 4), sharey=True)

# Likelihood as a density over theta (Beta(y+1, n-y+1) is the normalized binomial likelihood)
# (the binomial likelihood theta^y (1-theta)^(n-y), viewed as a density in theta and
#  renormalized, is exactly Beta(y+1, n-y+1) — i.e. the posterior under a flat Beta(1,1)
#  prior, which is why this curve coincides with the leftmost panel's posterior.)
lik = stats.beta.pdf(grid, y + 1, n - y + 1)

for ax, (label, a) in zip(axes, prior_strengths.items()):
    prior = stats.beta.pdf(grid, a, a)
    post  = stats.beta.pdf(grid, a + y, a + n - y)
    ax.plot(grid, prior, "--", color="gray", label="prior")
    ax.plot(grid, lik,   ":",  color="C1",   label="likelihood (data)")
    ax.plot(grid, post,  "-",  color="C0", lw=2, label="posterior")
    ax.axvline(mle, color="C1", alpha=0.4)     # MLE = 0.65
    ax.axvline(0.5, color="gray", alpha=0.4)   # prior mean
    ax.set_title(label, fontsize=9); ax.set_xlabel(r"$\theta$ (CTR)")
axes[0].legend(fontsize=8); axes[0].set_ylabel("density")
plt.tight_layout()
```

> 🩺 **What the four panels show.** Read them left to right as you turn up the prior-strength dial. In the leftmost panel ($\mathrm{Beta}(1,1)$, flat prior), the **posterior sits essentially on top of the likelihood**, peaked near the MLE 0.65 — the data are running the show. As you move right and the prior sharpens around 0.5, the gray dashed prior gets taller and narrower, and the solid blue posterior **slides leftward and tucks itself between** the likelihood (peaked at 0.65) and the prior (peaked at 0.5), always landing at the precision-weighted compromise. By the rightmost panel ($\mathrm{Beta}(40,40)$), the prior is so sharp that the posterior is **pinned near 0.5**, the data barely budging it. This single figure *is* the convex-combination formula of §3.1 made visual: the posterior is always trapped between prior and likelihood, and which one wins is set by their relative sharpness (precision). Sit with it for a second — when you later see a hierarchical model "shrink" a noisy group toward the grand mean (Chapter 10), it's doing exactly this, with the prior mean replaced by the across-group mean.

> 💡 **Intuition — the takeaway you carry forward.** Priors don't fight the data; they *negotiate* with it, and the terms of the negotiation are precision (= information). With lots of data the prior is a polite formality; with little data it's load-bearing. The skill isn't avoiding priors — it's calibrating their strength to how much you actually know, then *checking* that calibration with prior predictive simulation (Chapter 03) before trusting any tight interval the model hands you.

---

## ⚠️ Common errors & how to fix them

The traps in this chapter are mostly *silent* — your code runs, prints numbers, and lies. Learn the symptoms.

| Symptom | Cause | Fix |
|---|---|---|
| A "vague" `Normal(0, 0.001)` copied from a BUGS/JAGS model locks the parameter near 0 | BUGS parameterizes Normal by **precision**; PyMC by **SD**. `0.001` was meant as precision (SD ≈ 31.6), not SD. | Convert: SD $= 1/\sqrt{\text{precision}}$. In PyMC always pass **`sigma`** (a standard deviation). Re-read every transcribed prior. |
| Prior predictive (Ch 03) puts the outcome at ±40 SD; implied regression lines are absurd | Priors too wide for the (standardized) scale — "flat-ish" $\neq$ harmless | Standardize predictors/outcome first, then use unit-scale weakly-informative priors (`Normal(0,1)`). |
| Sampler explodes / posterior won't normalize; ESS near zero | A **flat/improper** prior (e.g. `pm.Flat`, or `Normal(0, 1e6)`) yields an improper posterior — common on variances or under logistic **separation** | Use a *proper* weakly-informative prior. For separation, a `Normal(0, 1.5)`-ish prior on logits tames it. |
| Group-level SD collapses to ≈0 and is wildly sensitive to a tiny constant | `InverseGamma(ε, ε)` on the **variance** $\sigma^2$ — the Gelman (2006) pathology | Put `pm.HalfNormal`, `pm.Exponential`, or `pm.HalfStudentT` on the **SD** $\sigma$ directly. Never `InvGamma(ε,ε)` on a hierarchical variance. |
| Heavy-tailed posterior, slow or stuck NUTS on a scale parameter | `HalfCauchy` tail too heavy for a weakly-informative likelihood | Switch to `HalfNormal`/`Exponential`, or `half-t(4)`; bump `target_accept=0.95` (Ch 07). |
| Posterior barely moves off the prior on a small dataset | Prior more *informative* than you intended | Run prior predictive + a sensitivity re-fit under a wider prior; widen if the conclusion is prior-driven (Ch 03). |
| `sample_prior_predictive(samples=...)` → `TypeError` | The `samples` argument was removed in PyMC v5 | Use **`draws=`** (`pm.sample_prior_predictive(draws=500, ...)`). |
| `LKJCholeskyCov` errors on `sd_dist`, or fitted correlations all ≈0 | `sd_dist` passed as a *named* RV / wrong shape; **or** `eta` set too large (concentrates on the identity) | Build `sd_dist` with `.dist()` (`pm.Exponential.dist(1., size=n)`, `shape[-1]==n`); lower `eta` toward 1–2. Remember: **larger eta ⇒ less correlation.** |
| You "chose" a prior because the algebra was conjugate, then it misbehaves | Conjugacy is convenience, not a modeling rationale | NUTS doesn't need conjugacy — choose the prior that fits your beliefs and samples well (e.g. drop `InvGamma` for `HalfNormal`). |
| A `Normal` prior put on a standard deviation / rate / lengthscale | Applied **location logic** to a **scale** parameter | Use a positive-support prior (`HalfNormal`/`Exponential`/`LogNormal`); §6.6. |

---

## 🧪 Exercises

Work these with code where it helps; they're designed to convert reading-knowledge into finger-knowledge.

**1. (Conceptual) Flat is not invariant.** Show by hand that a flat prior $p(\theta) \propto 1$ on $\theta \in (0, \infty)$ implies, after the change of variables $\phi = \log\theta$, the *non-flat* density $p(\phi) \propto e^{\phi}$. Then do it the other way: start flat on $\phi = \log\theta$ and show the implied $p(\theta) \propto 1/\theta$. *Hint:* multiply by $|d\theta/d\phi|$. **Bonus:** which of these is the Jeffreys prior for a scale parameter, and why is *that* the "right" noninformative choice?

**2. (Conjugate numbers) Gamma–Poisson by hand, then by sampler.** A call center logs $\{4, 7, 5, 6, 3\}$ calls in five hours. Using prior $\mathrm{Gamma}(s=2, r=1)$ (shape 2, rate 1), (a) write the posterior $\mathrm{Gamma}(\cdot, \cdot)$ and its mean by the §3.2 formula; (b) express the posterior mean as a convex combination of the prior mean and the MLE and report the prior's weight; (c) fit the same model in PyMC (`pm.Gamma` prior, `pm.Poisson` likelihood, `observed=` the five counts) and confirm `az.summary`'s posterior mean matches your (a) to within MC error. *Hint:* in PyMC, `pm.Gamma("lam", alpha=2, beta=1)` — `beta` is the rate.

**3. (Ridge/lasso) Draw the two priors and the two penalties.** On a single figure, plot the densities of $\mathrm{Normal}(0,1)$ and $\mathrm{Laplace}(0,1)$, and on a second figure plot their *negative log-densities* (the penalties $\beta^2/2$ and $|\beta|$, up to constants). Annotate: where does the Laplace's kink show up, and explain in one sentence why that kink — not the tail — is what produces exact-zero coefficients in MAP estimation. *Hint:* `scipy.stats.norm` and `scipy.stats.laplace`.

**4. (Precision footgun) Catch the bug.** A colleague ports a JAGS model to PyMC and writes `tau = pm.Normal("beta", mu=0, sigma=0.0001)` "to be vague, like the JAGS `dnorm(0, 0.0001)`." (a) What SD did they actually impose, and what did the JAGS author intend? (b) By what factor is the prior mis-scaled? (c) Rewrite it correctly. *Hint:* precision $= 1/\sigma^2$.

**5. (Scale priors) Reproduce the InverseGamma pathology, qualitatively.** Plot the densities of `InverseGamma(α=ε, β=ε)` on a standard deviation for $\varepsilon \in \{1, 0.1, 0.01, 0.001\}$ over $\sigma \in (0, 3)$, and overlay `HalfNormal(1)` and `Exponential(1)`. Describe how the IG mass piles up as $\varepsilon \to 0$ and argue, in two or three sentences, why this would spuriously shrink a hierarchical group SD toward 0. *Hint:* `scipy.stats.invgamma`. **Bonus:** read Gelman (2006) §3 and connect your picture to his Figure 1.

**6. (LKJ direction) Sample and confirm.** Using `pm.LKJCholeskyCov` with $n=2$ and `sd_dist = pm.Exponential.dist(1.0, size=2)`, draw from the prior with `pm.sample_prior_predictive(draws=2000)` for $\eta \in \{0.5, 1, 4\}$, extract the off-diagonal correlation, and plot its histogram for each $\eta$. Confirm that **larger $\eta$ concentrates correlations near 0** and $\eta < 1$ pushes them toward $\pm 1$. *Hint:* set `compute_corr=True` and read the `_corr` output; this is the Chapter 10 idiom in miniature.

---

## 📚 Resources & further reading

The canonical, *verified* sources behind this chapter — go deeper here, and treat the first two as required.

- **Gelman (2006), "Prior distributions for variance parameters in hierarchical models,"** *Bayesian Analysis* 1(3):515–534 — sites.stat.columbia.edu/gelman/research/published/taumain.pdf. The definitive case against `InverseGamma(ε,ε)` and for half-Cauchy/half-t on the SD. The most important single paper for §6.2/§6.6.
- **Stan Prior Choice Recommendations wiki** — github.com/stan-dev/stan/wiki/prior-choice-recommendations. The concrete 5-level hierarchy, the `normal(0,1)` / `student_t(3,0,1)` defaults, the scale-parameter advice, and the now-discouraged Cauchy default. The most practical page on the internet for §4 and §6.
- **Simpson, Rue, Riebler, Martins & Sørbye (2017), "Penalising model component complexity" (PC priors),** *Statistical Science* 32(1):1–28 — arXiv:1403.4630. The principled derivation behind §7; four-principle framework, base-model thinking.
- **Piironen & Vehtari (2017), "Sparsity information and regularization in the horseshoe and other shrinkage priors,"** *EJS* 11(2):5018–5051 — arXiv:1707.01694. Where the §5 sparsity story goes next; the regularized horseshoe (developed in Ch 14).
- **Austin Rochford, "The Hierarchical Regularized Horseshoe Prior in PyMC"** — austinrochford.com/posts/2021-05-29-horseshoe-pymc3.html. The canonical worked PyMC build for sparsity priors (ports to v5 mechanically).
- **Bayes Rules! Ch 5, "Conjugate Families"** — bayesrulesbook.com/chapter-5. Clean, learner-facing conjugate update formulas with the pseudo-count framing of §3.
- **John D. Cook, "A Compendium of Conjugate Priors"** — johndcook.com/CompendiumOfConjugatePriors.pdf. The exhaustive conjugate-pair table; bookmark it.
- **McElreath, *Statistical Rethinking* 2e, Ch 4** (PyMC port: github.com/pymc-devs/pymc-resources). The prior-predictive-for-height intuition and the "golem / small world" framing of §1.
- **Gelman, Carlin, Stern, Dunson, Vehtari & Rubin, *Bayesian Data Analysis* (BDA3), Ch 2–5** — stat.columbia.edu/~gelman/book/. The reference treatment of conjugacy, noninformative priors, weakly-informative priors, and hierarchical variance priors.
- **PyMC core notebook: "Prior and Posterior Predictive Checks"** — pymc.io/projects/docs/en/stable/learn/core_notebooks/posterior_predictive.html. The `standardize` + `sample_prior_predictive(draws=...)` idiom and the ±40-SD flat-prior cautionary tale (used in Ch 03).
- **PyMC API: `LKJCholeskyCov`** — pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.LKJCholeskyCov.html — verified `eta`, `n`, `sd_dist`, `compute_corr` params behind §6.4.
- **Kallioinen, Paananen, Bürkner & Vehtari (2023), "Detecting and diagnosing prior and likelihood sensitivity with power-scaling,"** *Stat. & Comp.* 34:57 — arXiv:2107.14054 (R package: n-kall.github.io/priorsense/). The modern, principled sensitivity method — concept here, practice in Ch 03. (No first-class Python port today; in Python, re-fit under alternative priors.)
- **Westfall (2017), Bambi default-priors paper** — arXiv:1702.01201. How Bambi auto-scales weakly-informative, data-scaled priors (rstanarm-style) — relevant when we lean on Bambi in Ch 09.

---

## ➡️ What's next

You now know what priors *are* — the objects, the families, the catalog, and the traps. But knowing the catalog is like owning a well-stocked toolbox: useless until you can pick the right tool for the job in front of you and *check* that you picked correctly. That's **Chapter 03 — Priors II: Choosing, Building & Checking.** There we make prior choice an *empirical, falsifiable* procedure rather than a vibe: the **prior predictive check** (simulate fake data from your priors and ask "is this a world I believe in?"), standardization and scaling so unit-scale priors mean the same thing everywhere, the schools of thought (subjective, objective, weakly-informative, empirical Bayes), the deep dive on scale/variance priors that §6.2 only sketched (HalfNormal vs HalfCauchy vs Exponential, and the full Gelman-2006 story), the LKJ and horseshoe families in action, the "should I look at the data first?" debate, and **prior sensitivity analysis** via power-scaling. Bring the catalog; we're about to put it to work.
