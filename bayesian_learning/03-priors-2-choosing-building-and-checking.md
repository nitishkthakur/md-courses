# Chapter 03 — Priors II: Choosing, Building & Checking

### *From "which distribution?" to a repeatable procedure for deciding whether your prior is any good.*

> *A note from me to you.*
>
> *In Chapter 02 you met the cast of characters — Normal, HalfNormal, Exponential, Beta, the LKJ, the horseshoe — and you learned the vocabulary of flat vs. proper, conjugate, weakly-informative, regularizing. That was the **catalog**. This chapter is the **craft**. Because here is the uncomfortable truth that nobody told me when I started: a prior is not something you look up. It is something you **build, simulate, and audit** — exactly the way you'd unit-test a piece of code before trusting it in production.*
>
> *The single most valuable habit in all of applied Bayesian work is the one I want to burn into your hands this chapter: before you ever touch the data with your likelihood, you push your priors **forward through the model** and look at the fake datasets they imply. This is the **prior predictive check**, and it is the cheapest, highest-leverage move in the entire workflow. I have watched experienced people argue for an hour about whether `Normal(0, 10)` is "reasonable" for a regression slope. Five lines of code and one plot ends the argument every time — because the model will happily tell you that those priors expect a person's height to vary by ±400 cm. You don't reason your way to good priors. You **simulate** your way there.*
>
> *We will also settle the philosophical fights — subjective vs. objective vs. weakly-informative vs. empirical Bayes, and the famous "am I allowed to look at the data first?" anxiety — not with dogma but with a working practitioner's answer. And we will face the parameters everyone gets wrong: the **scale parameters**, the standard deviations and variances, where a careless conjugate prior (`InverseGamma(ε, ε)`, I'm looking at you) can silently sabotage your hierarchical model. By the end you'll have a procedure, not a superstition.*

---

## What you'll be able to do after this chapter

- Run a complete **prior predictive check** in PyMC with `pm.sample_prior_predictive`, and read the implied-data and implied-regression-line plots to decide whether a prior is sane.
- **Iterate** priors deliberately — start vague, simulate, tighten, re-simulate — until the fake data your model dreams up looks like data that could plausibly exist in your domain.
- Explain *why* you **standardize predictors** so that a single unit-scale prior like `Normal(0, 1)` is sensible across all coefficients, and translate a slope prior between raw and standardized scales (in real cm-per-kg units).
- Choose **scale/variance priors** with judgment — HalfNormal vs. Exponential vs. HalfCauchy vs. half-Student-t — and articulate the Gelman (2006) argument for why `InverseGamma(ε, ε)` on a variance is a trap.
- Set the **LKJ** correlation prior's `eta` correctly (and remember which direction it shrinks), and understand the **horseshoe / regularized horseshoe** for sparse, many-predictor problems.
- Place yourself in the **schools of thought** (subjective, objective/reference, weakly-informative-as-default, empirical Bayes), give a defensible answer to "should I look at the data first?", and run a **manual prior sensitivity analysis** by refitting under alternative priors.

---

## 1. The central idea: a prior is a generative claim, so make it generate

Let me start with the mental shift that makes everything else click.

You are used to thinking of a prior as a *belief about a parameter*. That is true, but it is the wrong level of abstraction for choosing one. The problem is that parameters live in strange, abstract coordinate systems. What does it *feel* like for a regression slope to be `Normal(0, 10)`? You have no intuition for that, and neither do I, because slopes are not things we observe. We observe **data** — heights, counts, prices, click-rates. So the right question is never "is this a reasonable belief about $\beta$?" It is:

> **"If I believe this about my parameters, what kind of *datasets* am I claiming are plausible — before I've seen any data?"**

That question has a precise, computable answer. Your model is a generative recipe: priors feed parameters, parameters feed the likelihood, the likelihood emits data. If you run that recipe *forward* — sample parameters from the prior, then sample data from the likelihood given those parameters — you get the **prior predictive distribution**:

$$
p(\tilde y) = \int p(\tilde y \mid \theta)\, p(\theta)\, d\theta .
$$

In ASCII, for the skimmers:

```
prior predictive:   p(y~)  =  ∫  p(y~ | θ)  p(θ)  dθ
                              \_______/  \____/
                              likelihood  prior
```

This is the distribution over **fake data** your model can produce *with the likelihood and priors but no observations*. It is the model's imagination. And because it lives in data space — the same space your real observations live in — you can actually judge it. You know what a human height looks like. You know a reaction time can't be negative. You know a county's radon level isn't $10^{12}$. The prior predictive lets your hard-won domain knowledge enter through the front door, in units you understand.

> 💡 **Intuition:** A prior predictive check is a **dress rehearsal with no audience**. The model puts on the whole show — costumes (priors), choreography (the linear model), lights (the likelihood) — and you sit in the empty theater and ask: *does this look like a performance that could happen in the real world?* If the actors phase through walls and the building is 400 km tall, you fix the production *before* opening night (before you condition on data). It is so much cheaper to catch absurdity here than to discover later, when you run MCMC diagnostics (Chapter 07), that your posterior is fighting your priors.

> 📜 **Citation/Origin:** Prior predictive checks are central to the modern **Bayesian workflow** (Gelman et al. 2020, *Bayesian Workflow*, arXiv:2011.01808; Gabry, Simpson, Vehtari, Betancourt, Gelman 2019, *Visualization in Bayesian workflow*) and to Betancourt's *Towards A Principled Bayesian Workflow*. McElreath's *Statistical Rethinking* (2e, Ch. 4) popularized the specific "simulate the implied regression lines for height" exercise we'll rebuild below. We use it here as the **default decision procedure**, not an afterthought.

There is a second, subtler reason this matters, and it's the reason I trust prior predictive checks more than my own armchair reasoning. **Priors interact.** A prior on the intercept and a prior on the slope and a prior on the residual scale combine *nonlinearly* through the link function and the linear predictor. Each one can look individually innocent and the *combination* can be insane. You cannot eyeball the joint effect of three priors fed through an exponential link — but the forward simulation computes it for you exactly. This is especially vicious in GLMs (Chapter 09): a `Normal(0, 10)` prior on a logistic-regression coefficient sounds mild until you realize that, on the probability scale after the inverse-logit, it piles essentially all the prior mass at 0 and 1 — the model a priori "believes" every outcome is a near-certainty. Forward simulation catches this in one plot. Reasoning rarely does.

---

## 2. The prior predictive workflow in PyMC, end to end

Let's make this concrete and runnable. The PyMC function is `pm.sample_prior_predictive`. Here is the verified v5 signature and what comes back:

```python
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)

# pm.sample_prior_predictive(draws=500, var_names=None, random_seed=None,
#                            return_inferencedata=True, ...)
# -> returns an az.InferenceData with TWO groups:
#      idata.prior            : unobserved RVs (a, b, sigma, ...) + Deterministics
#      idata.prior_predictive : the observed RVs, sampled FORWARD (fake y)
```

> ⚠️ **Pitfall:** The keyword is **`draws`**, not `samples`. In old PyMC the argument was `samples`; it was removed. Calling `pm.sample_prior_predictive(samples=...)` raises a `TypeError` in v5. Use `draws=`. (The default is `draws=500`.)

The workflow has five steps, and you will repeat steps 2–4 until you're happy:

1. **Write the model** (priors + linear predictor + likelihood), with an `observed=` array so PyMC knows the shape and dtype of $y$. You don't need real $y$ values for the *prior* predictive — only the shape matters — but it's natural to pass your actual outcome array.
2. **Forward-sample** with `pm.sample_prior_predictive(draws=...)`.
3. **Plot the implied data** — `az.plot_ppc(idata, group="prior")` for the marginal distribution of fake $y$, and for regression problems, **overlay the implied regression lines** by drawing `a + b*x` for many prior draws.
4. **Judge it against domain knowledge.** Are the fake outcomes in a believable range? Do the implied relationships have plausible direction and steepness? Are there absurd values?
5. **Iterate.** If absurd, tighten (or occasionally loosen) the priors and go to 2.

Here is the canonical small example — a one-predictor linear regression on *standardized* variables, the pattern the PyMC core notebook uses to teach this exact lesson:

```python
def standardize(series):
    return (series - series.mean()) / series.std()

# (x_std, y_std are standardized predictor and outcome; see §3/§7 for the real data)
with pm.Model() as model_vague:
    a = pm.Normal("a", mu=0.0, sigma=10.0)        # DELIBERATELY too-wide intercept
    b = pm.Normal("b", mu=0.0, sigma=10.0)        # DELIBERATELY too-wide slope
    sigma = pm.Exponential("sigma", 1.0)
    mu = a + b * x_std
    pm.Normal("obs", mu=mu, sigma=sigma, observed=y_std)

    idata_vague = pm.sample_prior_predictive(draws=200, random_seed=RANDOM_SEED)

# What you now have:
idata_vague.prior["a"]            # 200 intercept draws
idata_vague.prior["b"]            # 200 slope draws
idata_vague.prior_predictive["obs"]   # 200 fake datasets in standardized-y units
```

> 🩺 **Diagnostic — reading `az.plot_ppc(idata_vague, group="prior")`:** This overlays many thin black curves (one per prior draw, each a kernel density of a *simulated* dataset) against a single thicker reference. For a sane prior on standardized data, those curves should bunch up roughly over $[-3, 3]$ — because standardized $y$ has SD 1, so almost all mass lives within a few SDs of zero. With `sigma=10` on `a` and `b`, you will instead see the fake $y$ spread across **±30 or ±40 standard deviations**. That is the model declaring, before seeing a single observation, that it would not be surprised if a person were 40 SDs taller than average. That is the absurdity you are hunting. Seeing it is the entire point.

> 🧮 **The math — why ±40 SD falls out of `Normal(0,10)` priors.** On standardized predictors, $\mathrm{Var}(x)=1$. The implied mean is $\mu = a + b x$, so $\mathrm{Var}(\mu) = \mathrm{Var}(a) + \mathrm{Var}(b)\,\mathrm{Var}(x) = 10^2 + 10^2\cdot 1 = 200$, giving $\mathrm{SD}(\mu)\approx 14$. The residual term adds $\mathbb{E}[\sigma^2]$; since $\sigma\sim\mathrm{Exponential}(1)$ here, $\mathbb{E}[\sigma^2] = \mathrm{Var}(\sigma)+\mathbb{E}[\sigma]^2 = 1 + 1 = 2$, so the marginal SD of simulated $y$ is about $\sqrt{200 + 2}\approx 14.2$ — and that residual contribution is negligible next to $\mathrm{Var}(\mu)=200$ anyway. So a 95%-ish prior predictive interval for $y$ is roughly $\pm 28$ standardized units, with the tails reaching ±40. Standardized outcomes that wander ±40 SD are not data; they're noise. **That single calculation is what a prior predictive check automates and visualizes for you.**

### Visualizing the implied regression *lines* (the McElreath move)

For regression, the marginal-$y$ plot is good but the **implied-lines** plot is better, because it shows the *relationship* the prior commits to, not just the spread. You draw the line $\mu = a + b x$ over the observed $x$-range for each prior draw and plot the bundle:

```python
xx = np.linspace(x_std.min(), x_std.max(), 50)

a_draws = idata_vague.prior["a"].values.flatten()      # shape (200,)
b_draws = idata_vague.prior["b"].values.flatten()

fig, ax = plt.subplots(figsize=(6, 5))
for ai, bi in zip(a_draws, b_draws):
    ax.plot(xx, ai + bi * xx, color="C0", alpha=0.2)
ax.set_xlabel("standardized predictor")
ax.set_ylabel("standardized outcome")
ax.set_title("Implied regression lines under the prior")
# A horizontal reference band at +/- 3 SD shows the plausible region for standardized y.
ax.axhline(3, ls="--", color="k"); ax.axhline(-3, ls="--", color="k")
```

> 🩺 **Diagnostic — reading the implied-lines plot.** With vague priors you'll get a **starburst**: lines firing off at every conceivable angle, many of them ludicrously steep, shooting far above +3 and below −3 within the observed $x$-range. A steepness of $|b|=10$ on standardized variables means "a one-SD change in $x$ moves the outcome by ten SDs" — a relationship so strong nothing in nature looks like it. A *good* prior produces a tidy fan of gently-sloped lines that mostly stay inside the ±3 band and don't commit to a sign. You are looking for **plausible, agnostic, not-yet-confident** — not flat, not crazy.

### Now tighten the priors and re-simulate

This is the iteration loop. On standardized data we have *physics* on our side: standardized variables have SD 1 by construction, so a slope much bigger than ~1–2 in absolute value is already a very strong claim. So `Normal(0, 1)` on the slope and intercept is a sane default; `Exponential(1)` keeps the residual scale near 1.

```python
with pm.Model() as model_good:
    a = pm.Normal("a", mu=0.0, sigma=0.5)     # intercept: standardized y ~ mean 0
    b = pm.Normal("b", mu=0.0, sigma=1.0)     # slope: |b|>~2 already a strong claim
    sigma = pm.Exponential("sigma", 1.0)
    mu = a + b * x_std
    pm.Normal("obs", mu=mu, sigma=sigma, observed=y_std)

    idata_good = pm.sample_prior_predictive(draws=200, random_seed=RANDOM_SEED)
```

Re-run the two plots. Now `az.plot_ppc(..., group="prior")` should show fake-$y$ curves living comfortably in $[-3, 3]$, and the implied-lines plot should be a neat agnostic fan. **That contrast — starburst vs. tidy fan — is the lesson.** You did not derive the "right" prior from a formula; you *found* it by simulating, looking, and adjusting. That is the craft.

> 🔧 **In practice:** Set `draws` to something cheap (50–500) for the prior predictive — it is forward sampling, essentially free, no MCMC. Bump it up only if the cloud looks too sparse to judge. And do this **before** you ever call `pm.sample`. The prior predictive check is your first line of defense; it costs seconds and saves hours.

---

## 3. Standardization: why unit-scale priors only make sense on standardized predictors

I keep saying "on standardized data, `Normal(0,1)` is sane." Let me now earn that claim, because it is the load-bearing convention for this entire course. **We standardize predictors by default, and from here on later chapters assume it.**

The argument is about *units*. A regression coefficient $\beta$ has units of `[outcome] / [predictor]`. So *the same numerical prior means wildly different things depending on the scales of your variables.* A prior is only as sensible as the units it lives in.

### The Howell example, in real units

Take the **Howell !Kung** dataset — adult height (cm) regressed on weight (kg). This is our recurring linear-regression example (introduced in Chapters 01–02; we'll reuse it for the full worked example in §9). Suppose you reach for what *sounds* like a mild, "let the data speak" slope prior on the **raw** scale:

$$
\beta_{\text{weight}} \sim \mathrm{Normal}(0,\ 10),
\qquad \text{units: cm of height per kg of weight.}
$$

What is that prior actually claiming? A slope of 10 cm/kg means that two people differing by a single kilogram differ, on average, by 10 cm in height. The prior puts a full SD of 10 on that, so values like 20 or 30 cm/kg are firmly inside the believable region. Now do the arithmetic: adult !Kung weights span roughly 30–60 kg. Under $\beta = 20$ cm/kg, going from the lightest to the heaviest adult would change predicted height by $30\text{ kg} \times 20\text{ cm/kg} = 600$ cm — **six meters**. The prior, before seeing data, is comfortable with a population whose tallest member is six meters taller than the shortest. That is not a weakly-informative prior. That is an *aggressively wrong* prior that merely *looks* humble because the number "10" is small and round.

> ⚠️ **Pitfall:** "Wide = humble" is a seductive lie. A wide prior on a coefficient is humble *about the parameter* but can be **extremely opinionated about the data** — it concentrates the prior predictive mass on impossible datasets. Humility is measured in data space, not parameter space. The only way to know is to forward-simulate.

Now **standardize**. Replace weight by $z = (\text{weight} - \overline{\text{weight}})/\,\mathrm{sd}(\text{weight})$. A coefficient on $z$ has units of `cm per (1 SD of weight)`. One SD of adult weight is roughly 6 kg. So a standardized slope of $b$ means: *a 6 kg increase in weight is associated with $b$ cm of height.* Adult height has an SD of about 7.7 cm. A standardized slope much bigger than, say, 2 would mean a 6 kg difference predicts a 15 cm height difference — already a very strong relationship. Hence:

$$
b \sim \mathrm{Normal}(0,\ 1) \quad\text{on standardized weight}
$$

says "a one-SD change in weight plausibly shifts height by something on the order of one height-SD, and I'd be mildly surprised by more than two." **That is a genuinely weakly-informative prior** — it rules out the absurd (six-meter ranges) while staying agnostic about sign and magnitude within the plausible band. The *same* `Normal(0,1)`, applied to the raw scale, would be a different and much worse claim. The number didn't change; the units did, and units are everything.

> 🧮 **The math — translating a slope between raw and standardized scales.** If $z_x = (x - \bar x)/s_x$ and (optionally) $z_y = (y - \bar y)/s_y$, and the standardized model is $z_y = a + b\,z_x + \varepsilon$, then on the raw scale the slope is
> $$
> \beta_{\text{raw}} = b \cdot \frac{s_y}{s_x}.
> $$
> For Howell with $s_y \approx 7.7$ cm and $s_x \approx 6.0$ kg, a standardized $b=1$ becomes $\beta_{\text{raw}} \approx 7.7/6.0 \approx 1.3$ cm/kg — a perfectly sensible "a kilo buys you about a centimeter" relationship, and worlds away from the 10–20 cm/kg the naive raw prior entertained. Keep this conversion in your back pocket: it lets you **set priors in standardized units (easy) and report effects in raw units (interpretable).** When only the predictor is standardized (outcome left in cm — common, because people want height in cm), then $\beta_{\text{raw}} = b / s_x$ and the intercept is interpreted at the mean predictor.

### Why standardization buys you so much

1. **One prior fits all coefficients.** After standardizing every predictor to SD 1, a single `Normal(0, 1)` (or `Normal(0, 2.5)` à la rstanarm, or `StudentT(3, 0, 1)` for heavier tails) is a reasonable default for *every* slope, regardless of whether the raw variable was measured in kilograms, dollars, or parts-per-billion. You stop hand-tuning a separate prior per predictor.

2. **The intercept becomes interpretable and gets an easy prior.** Center the predictors (subtract the mean) and the intercept $a$ is the expected outcome *at the average predictor values*. If you've also standardized the outcome, the intercept is near 0 with SD ~1, so `Normal(0, 0.5)` or `Normal(0, 1)` is natural. Without centering, the intercept is the outcome at $x=0$ — often a nonsensical extrapolation (a person of weight 0 kg), and a place where your prior has to be absurdly wide to compensate.

3. **The sampler is happier.** NUTS (Chapter 05) prefers parameters on comparable scales — a roughly isotropic, unit-scaled posterior geometry. Predictors on wildly different raw scales produce a stretched, correlated posterior that the sampler must fight, costing you effective samples and sometimes triggering divergences (Chapter 07). Standardization is partly a *computational* gift, not only an interpretive one.

> 🔧 **In practice — Gelman's "SD 0.5" refinement.** Gelman recommends scaling **continuous** predictors to mean 0 and SD **0.5** (not 1), and coding **binary** predictors as $\{-0.5, +0.5\}$ (or centered $\{0,1\}$). Why 0.5? It puts a continuous predictor's two-SD swing on the same footing as a binary predictor flipping from one category to the other, so a single coefficient prior is comparable across continuous and binary terms. This is the convention baked into **rstanarm** and **Bambi's** automatic priors. For teaching we usually standardize to SD 1 because the "one SD of $x$" interpretation is cleaner to say out loud; just know that the SD-0.5 convention exists and is what the formula-level tools assume. Either way, the headline rule stands: **standardize, then use unit-scale priors.**

> ⚠️ **Pitfall — leakage and "standardize with what?"** Compute the mean and SD used for standardization from your **training data only**, and store them, so you can apply the *same* transformation to new data at prediction time (Chapter 09 shows this with `pm.Data`). Standardizing test data with its own mean/SD leaks information and breaks the interpretation. Also: never standardize a variable you intend to model on its natural scale for a reason (e.g., a count you'll feed to a Poisson — you don't standardize the *outcome* of a count model; you standardize *predictors*).

> 🔧 **In practice — what a formula API actually picks for you (Bambi ↔ raw PyMC).** The "standardize, then unit-scale prior" rule is exactly what the formula tools automate. **Bambi** (`import bambi as bmb`) auto-generates weakly-informative, *data-scaled* priors in the rstanarm style — roughly `Normal(0, 2.5 · sd_y / sd_x)` on each slope (Westfall 2017, arXiv:1702.01201). You don't have to take this on faith; **build the model and read the priors back off it**:
>
> ```python
> import bambi as bmb
>
> model = bmb.Model("height ~ weight", data=adult)   # 'adult' from the Howell load in §9
> model.build()        # materialize the auto-priors
> print(model)         # <-- prints each term's auto Normal(0, scale) prior
> # idata = model.fit()  # would sample; we only want to INSPECT the priors here
> ```
>
> Because the scale factors drift across versions, *inspect* them rather than memorize a number. The raw-PyMC equivalent is just those same Normals written out by hand — there is no magic:
>
> ```python
> # Raw-PyMC equivalent of what Bambi built above (predictor standardized; outcome in cm).
> import numpy as np, pymc as pm
> sd_y = adult.height.std()
> with pm.Model() as model_pymc:
>     a = pm.Normal("Intercept", mu=adult.height.mean(), sigma=2.5 * sd_y)
>     b = pm.Normal("weight", mu=0.0, sigma=2.5 * sd_y / adult.weight.std())  # ~Bambi's auto-scale
>     sigma = pm.Exponential("sigma", 1.0 / sd_y)
>     mu = a + b * adult.weight.values
>     pm.Normal("obs", mu=mu, sigma=sigma, observed=adult.height.values)
> ```
>
> Same priors, two front doors — Bambi computes the scales from the data; raw PyMC makes them explicit. We use Bambi properly (and contrast it with hand-built PyMC) in **Chapter 09 (GLMs)**; here it's enough to see that "auto priors" are nothing but the scale-aware Normals this section argued for.

---

## 4. Priors for scale and variance parameters (where the bodies are buried)

Location parameters (means, intercepts, slopes) are forgiving. **Scale parameters** — standard deviations, residual scales, group-level SDs in hierarchical models — are where careless priors do real damage, and they are the most common cause of mysterious hierarchical-model pathologies. This section is the one I most want you to internalize.

A scale parameter $\sigma$ must be **positive**, so its prior lives on $(0, \infty)$. We have four serious candidates. Let me characterize each, then tell you what to actually use.

| Prior on $\sigma$ | PyMC | Tail behavior | Mass near 0 | Use when |
|---|---|---|---|---|
| **HalfNormal**$(0, s)$ | `pm.HalfNormal("σ", sigma=s)` | light (Gaussian) | finite, allows small σ | default for residual & group SD when you have a scale guess; lightest tails |
| **Exponential**$(\lambda)$ | `pm.Exponential("σ", lam)` | light (exp) | mode at 0, monotone decreasing | clean one-knob default; mean $=1/\lambda$; very common in PyMC examples |
| **HalfCauchy**$(0, s)$ | `pm.HalfCauchy("σ", beta=s)` | very heavy | finite | you genuinely won't rule out a large scale; Gelman-2006 default for group SD |
| **half-Student-t**$(\nu, 0, s)$ | half-t via `pm.StudentT`/`pm.HalfStudentT` *(verify spelling)* | tunable via $\nu$ | finite | compromise: heavier than HalfNormal, tamer than Cauchy; `ν=3–4` popular |
| ~~**InverseGamma**$(\varepsilon,\varepsilon)$ on $\sigma^2$~~ | `pm.InverseGamma` | — | **pathological** | ❌ avoid as a "noninformative" default — see Gelman 2006 below |

> 💡 **Intuition — what these knobs control.** Two things matter for a scale prior: **how much mass it puts near zero** (does it allow the group-level variation to be small or even effectively absent?) and **how heavy its right tail is** (does it allow the scale to occasionally be large?). HalfNormal and Exponential are *light-tailed* — they gently discourage very large scales. HalfCauchy is *heavy-tailed* — it lets $\sigma$ be large if the data insist, which is exactly what you want when you're unsure how much group-to-group variation there is and don't want the prior to strangle it. All four allow $\sigma$ to be small, which matters: in a hierarchical model the group-level SD might genuinely be near zero (groups barely differ), and your prior must not forbid that.

### Setting the scale by domain reasoning

Don't pick the hyperparameter `s` or `λ` by reflex. Pick it so the prior's bulk covers the scales you consider plausible. A clean rule:

- For a residual SD on **standardized** outcome, the residual SD is at most ~1 (you can't have more residual scatter than the total), so `pm.Exponential("sigma", 1.0)` (mean 1) or `pm.HalfNormal("sigma", 1.0)` is well-matched.
- For a residual SD on a **raw** outcome, set the scale to a generous fraction of the outcome's SD. Howell height has SD ~7.7 cm; `pm.Exponential("sigma", 1/10)` (mean 10 cm) or `pm.HalfNormal("sigma", 10)` says "typical residual scatter is on the order of several cm, could be 20, won't be 200."
- For a **group-level SD** in a hierarchical model (Chapter 10), think: *how much could group means plausibly differ?* If group means could differ by a handful of outcome-SDs at most, a HalfNormal or half-Cauchy with scale ~1 (on standardized outcome) covers it.

And — you guessed it — **verify the choice with a prior predictive check.** A scale prior is just as forward-simulable as any other; push it through and look at the spread of implied data.

### The Gelman (2006) argument: why *not* `InverseGamma(ε, ε)`

For decades the reflexive "noninformative" prior for a variance was the conjugate **inverse-Gamma**, $\sigma^2 \sim \mathrm{InvGamma}(\varepsilon, \varepsilon)$ with $\varepsilon$ tiny (0.001 was traditional, courtesy of BUGS/WinBUGS culture). It is conjugate for the Normal variance, so the posterior stays inverse-Gamma — algebraically tidy. **It is also a trap in hierarchical models,** and Gelman's 2006 paper is the canonical takedown.

> 📜 **Citation/Origin:** Gelman, A. (2006), *Prior distributions for variance parameters in hierarchical models*, **Bayesian Analysis 1(3): 515–534** (https://sites.stat.columbia.edu/gelman/research/published/taumain.pdf). The reference for *why* `InvGamma(ε,ε)` is bad and the **half-Cauchy on the SD** is the recommended weakly-informative default.

Here is the problem, stated plainly:

1. **There is no proper "noninformative" limit.** As $\varepsilon \to 0$, the `InvGamma(ε, ε)` prior does **not** converge to a sensible limiting prior; the limiting posterior for the group-level variance can be **improper**. So the comforting story — "$\varepsilon$ small means noninformative" — is false. There is no $\varepsilon=0$ to approach.

2. **It is secretly, strongly informative exactly where it hurts.** `InvGamma(ε, ε)` with small $\varepsilon$ has almost no density near $\sigma^2 = 0$ (it pushes the scale *away* from the boundary) but a sharp spike a little way out, and it is *very sensitive to $\varepsilon$*. In a hierarchical model where the true group-level SD is **small** or where you have **few groups** — precisely the regimes where you most need the prior to behave — the likelihood is weak and the arbitrary prior shape dominates. So depending on the data the inference can go either way: in the classic 8-schools setting it spuriously **holds $\tau$ away from 0** (the near-zero boundary density won't let it shrink as it should), while in other configurations the answer is simply driven by the arbitrary $\varepsilon$. The single robust description is **non-robustness**: the "noninformative" prior turns out to be the loudest voice in the room, and it's saying something arbitrary. (Contrast this with the *flat / improper* prior on the variance, whose characteristic failure mode is the opposite — letting the group SD **collapse toward 0**; see the diagnostic below.)

3. **Inference is not robust.** Change $\varepsilon$ from 0.001 to 0.01 and your posterior for the group SD can move materially. A prior whose conclusions hinge on a number you picked for algebraic convenience is the opposite of noninformative.

> 🩺 **Diagnostic — the symptom in the wild.** The shared fingerprint of a bad variance prior is **non-robustness**: the posterior for the group SD / $\tau$ moves materially under seemingly cosmetic prior changes. The two classic failure directions point to different culprits — `InvGamma(ε, ε)`'s near-zero boundary density tends to **hold $\tau$ artificially away from 0** (and makes the answer hostage to $\varepsilon$), whereas a **flat / improper** prior on the variance tends to let the **group SD collapse toward ~0**. Either way the fix is not "more tuning" — it is **change the prior**: put a half-Cauchy, half-t, HalfNormal, or Exponential on $\sigma$ *itself* (the SD), never an `InvGamma(ε,ε)` on $\sigma^2$ or a flat prior on the variance.

Gelman's positive recommendation: model the **SD** directly (not the variance, not the precision) and put a **weakly-informative half-t / half-Cauchy** on it. The half-Cauchy is finite and well-behaved near 0 (so small group SDs are allowed) but has a heavy enough tail to let the scale be large if the data demand it — it doesn't strangle genuine variation. For the 8-schools $\tau$, Gelman uses a half-Cauchy with a large scale (e.g., 25), which is "barely there" — it just rules out the truly absurd.

> 🔧 **In practice — my default recipe for scale priors:**
> - **Reach for `pm.HalfNormal` or `pm.Exponential` first.** They are light-tailed, single-parameter, sample beautifully, and are the right default when you have *any* sense of the plausible scale. PyMC's own example gallery uses `Exponential(1)` constantly for residual SDs on standardized data.
> - **Reach for half-Cauchy / half-t (`ν≈3–4`) when you genuinely expect the scale *might* be large** and refuse to rule it out — classically the **group-level SD in a hierarchical model with few groups**, where you don't want a light tail quietly shrinking the variation to zero.
> - **Mind the heavy tail's cost.** A `HalfCauchy` tail is *so* heavy it can slow NUTS and occasionally let the chain wander into huge-$\sigma$ regions, producing a few divergences or stuck transitions. If a half-Cauchy scale param is causing sampling grief, switch to half-t$(4)$ or HalfNormal, and/or raise `target_accept` to 0.95 (Chapter 07).
> - **Never** use `InvGamma(ε, ε)` on a variance as a "noninformative" default. If you want inverse-Gamma for conjugacy in a *non-hierarchical* known-form problem, fine — but with sensible, finite hyperparameters, not $\varepsilon \to 0$.

> 🧮 **The math — PC priors give the same recommendation from first principles.** The **Penalised Complexity (PC) prior** framework (Simpson, Rue, Riebler, Martins, Sørbye 2017, *Penalising model component complexity*, Statistical Science 32(1):1–28, arXiv:1403.4630) derives priors by penalizing departure from a simple **base model**. For a scale/variance parameter the natural base model is $\sigma = 0$ (the component is "off"); the PC prior penalizes the distance from that base at a constant rate, which yields an **Exponential prior on a transformed (KL-distance) scale** — concretely an exponential-type prior on $\sigma$ that decays from 0 with a single user-set rate. The punchline: the principled, decision-theoretic derivation lands you right back on **light-tailed, decreasing-from-zero priors on the SD with one interpretable knob** — i.e., the HalfNormal/Exponential family — vindicating Gelman's practical advice. (Exact PC closed forms are parameter-specific; re-derive from Simpson 2017 §2–3 for a particular case rather than guessing.)

---

## 5. The LKJ prior: priors on correlation matrices, and what `eta` really means

When you fit a model with **correlated** random effects — say, varying intercepts *and* varying slopes per group (Chapter 10), where a group's intercept and slope might covary — you need a prior over a **covariance matrix**. The clean way to do this is to decompose the covariance into scales and a **correlation matrix**, put scale priors (from §4) on the scales, and put an **LKJ prior** on the correlation matrix.

The LKJ distribution (Lewandowski, Kurowicka, Joe 2009) is a one-parameter family over valid $n\times n$ correlation matrices $R$ (symmetric, unit diagonal, positive-definite), with density

$$
p(R \mid \eta) \;\propto\; \det(R)^{\eta - 1}, \qquad \eta > 0 .
$$

The single shape parameter $\eta$ controls how much correlation the prior expects. This is the most-confused point in all of applied Bayesian modeling, so let me nail it with a small table and then a memory hook:

| $\eta$ | What it says about correlations | Analogy |
|---|---|---|
| $\eta = 1$ | **Uniform** over all valid correlation matrices | flat / `Beta(1,1)` on each correlation |
| $\eta > 1$ | Concentrates mass near the **identity** → correlations near 0 → **shrinks toward independence** | `Beta(2,2)`-like, more so as $\eta$↑ |
| $\eta < 1$ | Favors **strong** correlations (mass pushed toward ±1) | U-shaped |

> ⚠️ **Pitfall — the direction of `eta` is backwards from intuition.** People assume "bigger `eta` = more correlation." It is the **opposite**: **larger `eta` ⇒ *less* correlation** (more shrinkage toward the identity / independence). If your LKJ-equipped model reports all correlations near zero and you didn't expect that, your `eta` is probably too big. Conversely, $\eta=1$ is *not* "no prior" — it's uniform over correlation matrices, which in high dimensions actually favors weak correlations on its own (a geometric fact about the volume of correlation-matrix space).

> 🔧 **In practice:** For varying-slope models the standard mildly-regularizing default is **$\eta \approx 2$** — it gently pulls correlations toward zero (the `Beta(2,2)` analogue) without forbidding genuine correlation, which is usually exactly the right amount of skepticism. Use $\eta=1$ if you truly want to be uniform; reach below 1 only if you have a strong prior reason to expect large correlations.

### Building it in PyMC: `LKJCholeskyCov`

For sampling you almost always want the **Cholesky-parameterized** version, `pm.LKJCholeskyCov`. It is better-conditioned and faster than handling the full correlation matrix directly, because it works with the Cholesky factor $L$ of the covariance and lets NUTS move in an unconstrained space. Here is the verified v5 idiom:

```python
n = 2   # e.g. correlated (intercept, slope) per group

with pm.Model() as m:
    # sd_dist MUST be built with .dist() — an UNREGISTERED distribution, NOT a named RV.
    # It supplies the prior on the standard deviations of each component.
    sd_dist = pm.Exponential.dist(1.0, size=n)        # note .dist(), and size == n

    chol, corr, sigmas = pm.LKJCholeskyCov(
        "chol_cov",
        eta=2.0,            # mild shrinkage toward independence (Beta(2,2) analogue)
        n=n,
        sd_dist=sd_dist,
        compute_corr=True,  # default; returns the (chol, corr, stds) tuple
    )

    # Use chol directly in an MvNormal for the (intercept, slope) offsets:
    group_effects = pm.MvNormal("group_effects", mu=np.zeros(n), chol=chol,
                                dims=("group", "param"))   # dims illustrative
```

> ⚠️ **Pitfall — `sd_dist` shape and `.dist()`.** Two errors people hit: (1) passing a *named* RV (`pm.Exponential("s", 1.0)`) instead of the unregistered `pm.Exponential.dist(1.0, size=n)` — `LKJCholeskyCov` wants the `.dist()` form; (2) wrong size — `sd_dist` must have `shape[-1] == n` (scalars auto-broadcast). With `compute_corr=True` (the default) you get back `(chol, corr, stds)`, and PyMC stores `chol_cov_corr` and `chol_cov_stds` in the InferenceData so you can inspect the recovered correlations directly. If you instead set `compute_corr=False`, you get a *packed* lower-triangular vector that you must unpack with `pm.expand_packed_triangular(n, packed, lower=True)`.

There is also `pm.LKJCorr`, which returns the correlation matrix directly (no scales attached) — useful when you only need a correlation prior — but for the varying-effects covariance the Cholesky form is the recommended default. We'll lean on all of this heavily in **Chapter 10 (Hierarchical & Multilevel Models)**; here you just need to know the prior exists, that it has one knob `eta`, and which way that knob turns.

> 🩺 **Diagnostic — sanity-checking your LKJ prior.** You can (and should) prior-predictive-check the correlation prior too: sample from the model with no likelihood and plot the prior distribution of `chol_cov_corr[0, 1]` (the off-diagonal correlation). With $\eta=2$ you should see a symmetric bump centered at 0 with most mass in $(-0.7, 0.7)$ — agnostic but skeptical of extreme correlation. If it's flat to ±1, your `eta` is too small; if it's a tight spike at 0, too big.

---

## 6. The horseshoe and regularized horseshoe: priors that do variable selection

Sometimes you have **many** candidate predictors — dozens, hundreds, more than rows in some genomics problems — and you believe that **most coefficients are essentially zero** but a few are genuinely important. You want a prior that aggressively shrinks the noise coefficients to zero while letting the real signals stay large and (mostly) unshrunk. A single global `Normal(0, τ)` (ridge / L2) can't do this: pick $\tau$ small and you crush the real signals; pick it large and you fail to shrink the noise. You need a prior that is **adaptive per coefficient**. That is the **horseshoe**.

> 💡 **Intuition — a continuous spike-and-slab.** The ideal sparsity prior is "spike-and-slab": a spike of probability exactly at zero (kill the noise) plus a diffuse slab (let signals roam). That's discrete and combinatorially nasty to sample. The horseshoe is its smooth, gradient-friendly cousin: a global knob that pulls *everything* toward zero, plus a *local* per-coefficient knob with such a heavy tail that any coefficient with real signal can "punch through" the global shrinkage and escape. Most coefficients ride the global pull down to ~0; a few break free. Hence "horseshoe" — the implied shrinkage weight has a density shaped like a horseshoe, piling up at "fully shrunk" and "not shrunk."

> 🧮 **The math — the original horseshoe** (Carvalho, Polson & Scott 2010, *Biometrika* 97(2):465–480; AISTATS precursor 2009):
> $$
> \beta_j \sim \mathrm{Normal}(0,\ \tau^2 \lambda_j^2), \qquad
> \lambda_j \sim \mathrm{HalfCauchy}(0, 1), \qquad
> \tau \sim \mathrm{HalfCauchy}(0, \tau_0).
> $$
> $\tau$ is the **global** scale (shrinks all coefficients); $\lambda_j$ is the **local** scale (the heavy half-Cauchy tail lets coefficient $j$ escape). The product $\tau\lambda_j$ is the effective SD of $\beta_j$.

The plain horseshoe has two practical weaknesses, which the **regularized horseshoe** fixes:

> 📜 **Citation/Origin:** Piironen & Vehtari (2017), *Sparsity information and regularization in the horseshoe and other shrinkage priors*, **Electronic Journal of Statistics 11(2): 5018–5051** (arXiv:1707.01694). Two fixes:
> 1. **Set the global scale from a guess of how many coefficients are nonzero.** Instead of picking $\tau_0$ blind, derive it from a prior guess $p_0$ of the number of nonzero coefficients among $p$ total: $\tau_0 = \dfrac{p_0}{p - p_0}\cdot \dfrac{\sigma}{\sqrt{n}}$. This injects your actual sparsity belief, rather than leaving the global scale to flap in the wind.
> 2. **Add a finite-width slab so large coefficients are regularized.** Plain horseshoe leaves the truly large coefficients with *infinitely* heavy tails (unregularized), which can hurt. The regularized version caps the local scale by a slab of width $c$: $\tilde\lambda_j^2 = \dfrac{c^2 \lambda_j^2}{c^2 + \tau^2\lambda_j^2}$, with $c^2$ getting its own prior (e.g. inverse-Gamma). For small $\beta_j$ this is ≈ the horseshoe; for large $\beta_j$ it gracefully reverts to a Normal-with-SD-$c$ slab — finite, well-behaved tails.

> 🔧 **In practice — when to reach for it, and how to sample it.** Use the (regularized) horseshoe when (a) you have **many predictors**, (b) you expect **genuine sparsity** (most effects ≈ 0), and (c) you want the model to do the selecting rather than you hand-pruning. It is overkill for a handful of predictors — there a plain weakly-informative `Normal(0,1)` is simpler and fine. Crucially, **always reparameterize the horseshoe non-centered** (Chapter 07's centered-vs-non-centered idea) — the multiplicative, heavy-tailed hierarchy creates a brutal funnel geometry that NUTS cannot navigate in the naive centered form. Austin Rochford's *The Hierarchical Regularized Horseshoe Prior in PyMC* (https://austinrochford.com/posts/2021-05-29-horseshoe-pymc3.html) is the canonical worked PyMC build; porting it to v5 is mechanical. We revisit sparsity priors in depth in **Chapter 14 (Nonparametric & Flexible Models)**.

---

## 7. Schools of thought: where do priors *come from*?

You now know how to build and check priors. But people will ask you — sometimes pointedly, in a code review or a paper rebuttal — *"on what philosophical basis did you choose that prior?"* There are four recognizable answers, and a mature practitioner understands all four and knows which one they're using.

**1. Subjective Bayes.** The prior encodes *your* genuine beliefs, elicited carefully, and the whole point is that it's personal and informative. This is the original, philosophically purest position: probability is degree of belief, full stop. In practice, true subjective priors show up when you have real expert knowledge worth encoding — a pharmacologist's prior on a dose-response slope, a physicist's prior on a rate constant. The strength: you use everything you know. The cost: reproducibility and the "whose beliefs?" critique. Subjective priors are powerful *and* you must disclose and defend them.

**2. Objective / reference Bayes.** The dream of a prior that "lets the data speak" — Jeffreys priors, reference priors, maximum-entropy priors. The aspiration is a canonical, belief-free prior so two analysts with the same data and model reach the same posterior. Jeffreys priors ($p(\theta)\propto\sqrt{\det I(\theta)}$) have the lovely property of **reparameterization invariance**. But — as Chapter 02 discussed and §4 hammered — these priors are often **improper**, can yield improper posteriors silently (especially on hierarchical variances and under separation in logistic regression), and break prior predictive checks and marginal-likelihood methods. "Objective" is a noble goal and a fragile foundation for applied multilevel work.

**3. Weakly-informative-as-default (the pragmatic middle, and our house style).** Don't try to encode everything you believe, and don't pretend to believe nothing. Instead, choose a prior that **rules out the absurd and stays agnostic within the plausible** — `Normal(0, 1)` on standardized coefficients, HalfNormal/Exponential on scales, LKJ($\eta=2$) on correlations. It provides gentle regularization (better-conditioned posteriors, no wild estimates under separation or small $n$) without imposing strong domain claims. This is the position of the modern PyMC/Stan ecosystem and of this course.

> 📜 **Citation/Origin — the Stan Prior-Choice hierarchy.** The Stan team's *Prior Choice Recommendations* wiki (https://github.com/stan-dev/stan/wiki/prior-choice-recommendations) lays out an explicit ladder, from worst to best-for-most:
> 1. **Flat / improper** — *not recommended.*
> 2. **Super-vague but proper**, e.g. `normal(0, 1e6)` — *not recommended* (all the downsides of flat, no upside).
> 3. **Weakly informative, generously wide**, e.g. `normal(0, 10)` — okay-ish, often still too wide (recall Howell).
> 4. **Generic weakly informative** — **`normal(0, 1)`** (Gelman) or **`student_t(3, 0, 1)`** (Vehtari, heavier tails for robustness to occasional large effects) on standardized predictors. **This is the recommended default.** rstanarm's default is `normal(0, 2.5)`; for coefficients on predictors scaled to SD 0.5, `student_t(ν, 0, s)` with $3 < \nu < 7$ is current guidance.
> 5. **Specific informative** — when you actually have the knowledge.
>
> Two historical notes the wiki makes: the old **Cauchy** coefficient prior (`student_t(1, 0, 2.5)`) is now **discouraged** — tails so heavy they cause sampling trouble when the data are only weakly informative; and **`inv_gamma(0.4, 0.3)`** is suggested for the Negative-Binomial dispersion $\phi$, not the `inv_gamma(ε, ε)` you must avoid for variances.

**4. Empirical Bayes.** Estimate the prior's hyperparameters **from the data** — e.g., fit the data, find the group-level SD that maximizes the marginal likelihood, then plug that point estimate back in as if it were a known prior. It's a fast, often-good approximation to a full hierarchical model and shows up all over machine learning (it's what a lot of "automatic regularization" really is). The catch: by using the data to set the prior and then conditioning on the *same* data, you **double-count** and **understate uncertainty** — the plugged-in hyperparameter is treated as known when it isn't. The honest Bayesian alternative is to put a **hyperprior** on those hyperparameters and let the model learn them *with* their uncertainty — i.e., a full hierarchical model (Chapter 10). Empirical Bayes is the MAP-shaped shortcut; full Bayes integrates.

### "Am I allowed to look at the data first?" — the debate, resolved

This is the anxiety I promised to settle. The purist worry: *if I peek at the data before setting my prior, haven't I cheated — used the data twice, biased my inference, violated the whole "prior comes before data" story?*

Here is the working practitioner's answer, in layers:

- **Looking at the data to choose the *model* and the *scales* of your variables is normal and fine.** You standardize predictors using the data's mean and SD — that's "using the data," and nobody objects, because it's a *modeling* decision about units, not a smuggling of the outcome into the prior. Likewise, knowing that heights are around 150–180 cm is domain knowledge that happens to also be visible in the data; using it to avoid a six-meter prior is just sanity.

- **Using the *outcome* to tune your prior toward the answer you want is cheating**, and it does double-count — that's the empirical-Bayes pitfall above, taken to a bad place. If you fit, see the data prefer $\beta\approx 2$, then "set" a tight prior at 2 and re-fit, you've manufactured false confidence.

- **The weakly-informative middle dissolves most of the tension.** Because a weakly-informative prior is calibrated to the *scale* of the problem (which you legitimately know) rather than the *value* of the effect (which you don't), looking at the data to pick the scale doesn't bias your conclusions about the effect. You're using the data to get the *units* right, not the *answer* right. This is why standardization-plus-unit-priors is so freeing: it lets you set good priors *with* a glance at the data and a clear conscience.

- **And whatever you do, prior predictive checks and sensitivity analysis are your insurance.** If you're worried a prior is too informative, *check whether it matters* (§8). If the posterior is dominated by the likelihood, the philosophical fuss is moot. If it isn't, you've found exactly the prior you need to think hard about and disclose.

> 💡 **Intuition:** The "don't look at the data" rule is really "don't use the *outcome* to pre-load the *answer*." Using the data to fix **units, scales, and model structure** is part of responsible modeling — it's the difference between calibrating your instrument and faking your reading. Standardize freely; choose weakly-informative priors freely; never tune a prior toward the effect estimate you're hoping to publish.

---

## 8. Prior sensitivity analysis: does the prior actually matter?

The final discipline. You've built priors, checked them forward, and chosen weakly-informative defaults. Before you trust the posterior, ask the question that separates careful work from hopeful work:

> **"If I had chosen a *different* reasonable prior, would my conclusions change?"**

If the answer is "no" — the posterior is essentially the same under several reasonable priors — then the data are driving the inference and you can relax about the prior. If the answer is "yes" — the conclusion flips with a plausible prior change — then your result is **prior-dependent**, you must say so loudly, and the prior becomes a thing you genuinely need to justify (and the data alone can't settle the question). Either outcome is valuable; the failure is not checking.

> 📜 **Citation/Origin — the modern method, and why we do it by hand.** The principled, automatic approach is **power-scaling sensitivity** (Kallioinen, Paananen, Bürkner, Vehtari 2023, *Detecting and diagnosing prior and likelihood sensitivity with power-scaling*, Statistics & Computing 34:57, arXiv:2107.14054), implemented in the R package **priorsense** (`powerscale_sensitivity`). It perturbs the prior (and likelihood) by raising them to a power $\alpha$ and uses importance sampling to measure how much the posterior moves — no refitting required. The catch for us: **there is no first-class Python/PyMC port** I can point you to with confidence. So in Python we do the robust, low-tech thing: **refit the model under a few alternative priors and compare the posteriors directly.** (If you want the power-scaling spirit without R, you can reweight existing draws by the log-prior ratio via importance sampling — but a clean manual refit is clearer for teaching and safer than untrusted importance weights.)

### The manual refit recipe

```python
# pseudocode-ish: a reusable harness for prior sensitivity by refitting

def fit_with_priors(x_std, y_std, b_sigma, sigma_lam, seed=RANDOM_SEED):
    """Fit the same regression under a chosen slope-prior SD and residual-rate."""
    with pm.Model() as m:
        a = pm.Normal("a", 0.0, 0.5)
        b = pm.Normal("b", 0.0, b_sigma)              # <-- the prior we vary
        sigma = pm.Exponential("sigma", sigma_lam)    # <-- and this one
        mu = a + b * x_std
        pm.Normal("obs", mu=mu, sigma=sigma, observed=y_std)
        idata = pm.sample(1000, tune=1000, chains=4,
                          target_accept=0.9, random_seed=seed)
    return idata

# Fit under a grid of "reasonable" priors:
priors_to_try = {
    "tight   b~N(0,0.5)": dict(b_sigma=0.5, sigma_lam=1.0),
    "default b~N(0,1)":   dict(b_sigma=1.0, sigma_lam=1.0),
    "wide    b~N(0,5)":   dict(b_sigma=5.0, sigma_lam=1.0),
    "wide-σ  Exp(0.1)":   dict(b_sigma=1.0, sigma_lam=0.1),
}
fits = {name: fit_with_priors(x_std, y_std, **kw) for name, kw in priors_to_try.items()}

# Compare the posterior for the slope b across priors:
az.plot_forest([fits[n] for n in priors_to_try],
               model_names=list(priors_to_try),
               var_names=["b"], combined=True)

for name, idata in fits.items():
    # az.summary returns a one-row DataFrame for "b"; .loc gives clean scalars.
    # NOTE: hdi_3% / hdi_97% are the DEFAULT 94% HDI bounds. If you pass
    # hdi_prob=0.90 the column names change to hdi_5% / hdi_95% — update accordingly.
    row = az.summary(idata, var_names=["b"]).loc["b", ["mean", "hdi_3%", "hdi_97%"]]
    print(name, row.round(2).to_dict())
```

> 🩺 **Diagnostic — reading the sensitivity forest plot.** `az.plot_forest` with `var_names=["b"]` across the refits draws one interval per prior, stacked. **If the intervals for `b` essentially overlap** — same center, similar width — your slope is **robust** to the prior; the likelihood dominates and you can report the default-prior result with confidence. **If they march left/right or balloon as the prior widens**, you have a **prior-sensitive** estimate — likely a sign the data are weak relative to the prior for that parameter. Two honest moves then: (1) report the sensitivity explicitly ("the slope estimate ranges from X to Y across reasonable priors"), and (2) decide whether you actually *have* the domain knowledge to justify a tighter prior — if you do, use it and say so; if you don't, your data simply can't pin the effect down, which is itself the finding.

> 🔧 **In practice:** Don't sensitivity-test *everything* by reflex — focus on the priors that (a) are doing real regularization work, or (b) sit on parameters where the data are thin. Group-level SDs in hierarchical models with few groups (§4!) and coefficients in small-$n$ regressions are the usual suspects. And remember the ordering: **prior predictive check first** (is the prior even sane *a priori*?), then fit, then **prior sensitivity** (does the choice change the posterior?). The two checks answer different questions — "is this prior plausible?" vs. "does this prior matter?" — and you want both.

---

## 9. Worked example: building & prior-predictive-checking a regression on standardized Howell

Time to put it all together on real data, the way you'll actually work. We'll regress **height on weight** for adult !Kung from the Howell dataset, standardize, build a *bad* prior and a *good* prior, prove the difference with implied-regression-line plots, then briefly fit and sensitivity-check. This is the canonical McElreath exercise, done the modern PyMC/ArviZ way.

### Load and standardize

```python
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)

url = "https://raw.githubusercontent.com/rmcelreath/rethinking/master/data/Howell1.csv"
df = pd.read_csv(url, sep=";")          # columns: height, weight, age, male
adult = df[df.age >= 18].copy()         # restrict to adults: weight->height is ~linear here

# Standardize predictor and outcome (store the stats so we can convert back to cm/kg)
w_mean, w_sd = adult.weight.mean(), adult.weight.std()
h_mean, h_sd = adult.height.mean(), adult.height.std()
adult["weight_s"] = (adult.weight - w_mean) / w_sd
adult["height_s"] = (adult.height - h_mean) / h_sd

x_std = adult.weight_s.values
y_std = adult.height_s.values
print(f"n = {len(adult)},  weight SD = {w_sd:.1f} kg,  height SD = {h_sd:.1f} cm")
# Roughly: n ~ 352, weight SD ~ 6.5 kg, height SD ~ 7.7 cm
# (§3 uses the round figure s_x ~ 6 kg for the conversion intuition; the actual
#  adult-sample SD is ~6.5 kg, which is what lands the raw slope at ~0.9 cm/kg below.)
```

We standardized *both* variables here for the cleanest "unit-scale priors" story; in practice you might keep height in cm so the intercept reads as "mean height" and the slope reads as cm-per-SD-of-weight. Either is fine — just be consistent and remember the conversion from §3.

### Two models: a deliberately bad prior and a good one

```python
# --- BAD: vague priors that SOUND humble but imply absurd data ---
with pm.Model() as model_bad:
    a = pm.Normal("a", 0.0, 10.0)        # intercept SD 10 on standardized height (!!)
    b = pm.Normal("b", 0.0, 10.0)        # slope SD 10: "1 SD of weight -> +/-10 SD of height"
    sigma = pm.Exponential("sigma", 1.0)
    mu = a + b * x_std
    pm.Normal("obs", mu=mu, sigma=sigma, observed=y_std)
    prior_bad = pm.sample_prior_predictive(draws=200, random_seed=RANDOM_SEED)

# --- GOOD: weakly-informative, scale-aware priors ---
with pm.Model() as model_good:
    a = pm.Normal("a", 0.0, 0.5)         # standardized height ~ mean 0, SD 1; SD 0.5 is plenty
    b = pm.Normal("b", 0.0, 1.0)         # |b|>~2 already a very strong height~weight relation
    sigma = pm.Exponential("sigma", 1.0) # residual SD ~ 1 on standardized outcome
    mu = a + b * x_std
    pm.Normal("obs", mu=mu, sigma=sigma, observed=y_std)
    prior_good = pm.sample_prior_predictive(draws=200, random_seed=RANDOM_SEED)
```

### The reveal: implied regression lines, bad vs. good

```python
xx = np.linspace(x_std.min(), x_std.max(), 50)

fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
for ax, prior, title in [(axes[0], prior_bad, "BAD: Normal(0,10) priors"),
                         (axes[1], prior_good, "GOOD: Normal(0,1) priors")]:
    a_d = prior.prior["a"].values.flatten()
    b_d = prior.prior["b"].values.flatten()
    for ai, bi in zip(a_d, b_d):
        ax.plot(xx, ai + bi * xx, color="C0", alpha=0.15)
    ax.axhline(3, ls="--", color="k"); ax.axhline(-3, ls="--", color="k")
    ax.set(title=title, xlabel="standardized weight",
           ylim=(-6, 6))
axes[0].set_ylabel("standardized height")
```

> 🩺 **Diagnostic — what you'll see.** The **left (bad)** panel is a chaotic starburst: lines at every angle, most of them leaving the ±3 SD band almost immediately, plenty of them sloping *downward* (heavier people are shorter?) as confidently as upward, and many so steep that a one-SD weight change implies a ten-SD height change. Translate the worst of these back with $\beta_{\text{raw}} = b\cdot s_y/s_x \approx b\cdot 7.7/6.0$: a prior slope of $b=10$ is ~13 cm/kg, i.e. a 30 kg weight range maps to ~390 cm of height range — that *starburst is the six-meter person from §3, drawn*. The **right (good)** panel is a tidy fan of gently positive-and-negative lines that mostly respect the ±3 band — agnostic about the sign and steepness, but committed to "this is a relationship between two normal-sized human quantities." **That visual contrast is the entire lesson of the chapter in one figure.**

You can also look at the marginal of the implied outcome:

```python
az.plot_ppc(prior_bad, group="prior")    # fake heights spanning tens of SDs -> absurd
az.plot_ppc(prior_good, group="prior")   # fake heights bunched in [-3, 3] -> plausible
```

> 🩺 **Diagnostic — `az.plot_ppc(..., group="prior")`.** Each thin curve is the KDE of one *simulated* dataset; the spread of curves is the prior predictive spread. Bad priors smear the simulated standardized heights across ±30–40 SD; good priors keep them in a believable few-SD window. (If you kept height in raw cm, the good prior should produce simulated heights mostly in, say, 120–200 cm, never negative, never 4 m.)

### Now fit the good model and confirm the data agree

```python
with model_good:
    idata = pm.sample(1000, tune=1000, chains=4, target_accept=0.9,
                      random_seed=RANDOM_SEED)
    # posterior predictive for later checks (Chapters 08/11):
    idata = pm.sample_posterior_predictive(idata, extend_inferencedata=True,
                                           random_seed=RANDOM_SEED)

az.summary(idata, var_names=["a", "b", "sigma"])
```

> 🩺 **Diagnostic — reading the summary.** You're looking for $\hat R \le 1.01$ and `ess_bulk`, `ess_tail` comfortably above ~400 (Chapter 07 defines these; this clean, well-conditioned standardized model should sail through with **0 divergences**). The slope `b` should land strongly positive — around **0.72–0.78** on the standardized scale for adult Howell, an extremely tight posterior because $n\approx 352$ and the height–weight relationship is strong. (When you standardize *both* variables, the slope just *is* the Pearson correlation $r(\text{weight},\text{height})$, which for adult Howell is $\approx 0.75$ — that's why it lands there.) Convert it to raw units to report. With the *full* adult sample the weight SD is closer to $s_x\approx 6.5$ kg, so $\beta_{\text{raw}} = b\cdot s_y/s_x \approx 0.75 \times 7.7/6.5 \approx 0.9$ cm/kg — **"about nine-tenths of a centimeter of height per kilogram of weight,"** the canonical McElreath result and exactly the human-scale answer the good prior's implied lines anticipated and the bad prior's starburst buried in noise.

> 💡 **The payoff.** Notice that with $n\approx 352$ and a strong relationship, the **posterior is essentially the same whether you used the good prior or the (sane parts of the) bad prior** — the likelihood dominates. So was the prior predictive check a waste? Absolutely not. First, you didn't *know* the data would dominate until you checked (with small $n$, or a weak relationship, or a GLM link, the vague prior can absolutely pollute the posterior). Second, the bad priors make the **sampler** work harder and can manufacture divergences in more complex models. Third — and this is the habit — you want the check to be **automatic**, so that on the day the prior *does* matter, you catch it. Cheap insurance you buy every single time.

### Prior sensitivity in one glance

```python
def fit_b(b_sigma):
    with pm.Model() as m:
        a = pm.Normal("a", 0.0, 0.5)
        b = pm.Normal("b", 0.0, b_sigma)
        sigma = pm.Exponential("sigma", 1.0)
        pm.Normal("obs", mu=a + b * x_std, sigma=sigma, observed=y_std)
        return pm.sample(1000, tune=1000, chains=4, target_accept=0.9,
                         random_seed=RANDOM_SEED, progressbar=False)

fits = {f"b~N(0,{s})": fit_b(s) for s in (0.5, 1.0, 5.0)}
az.plot_forest(list(fits.values()), model_names=list(fits),
               var_names=["b"], combined=True)
```

> 🩺 **Diagnostic.** The three slope intervals will sit essentially on top of one another — robust. With $n=352$ the prior on `b` barely matters. Run the same harness on a **subsample of 10 adults** and watch the intervals separate: that's the regime where the prior predictive check and sensitivity analysis stop being academic and start being load-bearing. Sit with that — *the smaller and weaker your data, the more your priors do, and the more these checks earn their keep.*

---

## ⚠️ Common errors & how to fix them

| Symptom | Cause | Fix |
|---|---|---|
| `sample_prior_predictive(samples=...)` raises `TypeError` | The `samples` argument was removed in PyMC v5 | Use `draws=` (default `draws=500`) |
| Prior predictive `obs` ranges to ±30–40 SD; nonsensical fake data | Priors too wide for the (standardized) scale | Tighten to unit-scale weakly-informative; standardize predictors (and outcome) first, then forward-simulate again |
| Implied regression lines are a steep "starburst" at every angle | Slope prior SD far too large for standardized variables ($|b|\gg 2$ is already extreme) | Drop slope prior to `Normal(0, 1)` (or `StudentT(3,0,1)`); re-plot the implied lines |
| A `Normal(0,10)` coefficient looks "mild" but breaks logistic/Poisson models | On a link scale (inverse-logit / exp) a wide coefficient piles prior mass at the extremes | **Always** prior-predictive-check on the *outcome/probability* scale, not the parameter scale (Ch 09) |
| Coefficients on raw predictors need wildly different priors per term | Predictors on very different raw scales (kg vs. \$ vs. ppb) | Standardize all predictors (mean 0, SD 0.5 or 1) so one unit-scale prior fits every coefficient |
| Group-level SD collapses to ~0; posterior wildly sensitive to tiny prior tweaks | `InverseGamma(ε,ε)` (or flat/improper) prior on a **variance** — the Gelman-2006 pathology | Put `pm.HalfNormal` / `pm.Exponential` / half-t / half-Cauchy on the **SD** $\sigma$ itself, never `InvGamma(ε,ε)` on $\sigma^2$ |
| Heavy-tailed, slow, or stuck NUTS on a scale parameter; stray divergences | `HalfCauchy` tail too heavy relative to a weak likelihood | Switch to `HalfNormal`/`Exponential` or half-t$(4)$; raise `target_accept` to 0.95 (Ch 07) |
| `LKJCholeskyCov` errors on `sd_dist` shape/type | Passed a *named* RV, or `sd_dist` shape $\neq n$ | Build `sd_dist` with `.dist()` (e.g. `pm.Exponential.dist(1., size=n)`) and ensure `shape[-1]==n` |
| LKJ model reports all correlations ≈ 0 unexpectedly | `eta` too large (concentrates mass on the identity / independence) | Lower `eta` toward 1–2; remember **larger `eta` ⇒ *less* correlation** |
| Posterior barely moves off the prior on small data | Prior unintentionally too informative for the available data | Run a prior predictive check *and* a manual prior sensitivity refit; widen if the prior is doing more work than you intended |
| "Improper posterior" / sampler explodes / `NaN` logp | Flat/improper prior on a variance, or separation in logistic regression | Use a *proper* weakly-informative prior on every scale parameter and coefficient |
| You "looked at the data" and feel you cheated | Confusing *using data for units/structure* with *using the outcome to pre-load the answer* | Standardizing and choosing scale-aware weakly-informative priors is fine; never tune a prior toward the effect estimate you want |

---

## 🧪 Exercises

**1. (Conceptual — units.)** A colleague models reaction time (seconds, mean ≈ 0.4 s) on caffeine dose (mg, range 0–400) with the *raw* prior $\beta \sim \mathrm{Normal}(0, 1)$ s/mg. Without coding, describe the implied effect of going from 0 to 400 mg, and explain why this prior is *not* weakly informative despite the small SD. Then state the standardized prior you'd use and convert it back to s/mg using $\beta_{\text{raw}} = b\,s_y/s_x$ (assume $s_y = 0.1$ s, $s_x = 100$ mg). *Hint: 400 mg is 4 SDs of dose; what does a slope of 1 s/mg do over that range?*

**2. (Coding — the iteration loop.)** Generate synthetic data: `x = rng.normal(size=80); y = 1.5 + 2.0*x + rng.normal(scale=1, size=80)`, then standardize both. Build three regression models with slope priors `Normal(0, 0.1)`, `Normal(0, 1)`, and `Normal(0, 20)`. For each, run `pm.sample_prior_predictive(draws=200)` and plot the implied regression lines. Identify which is too tight (forbids the true slope), which is sane, and which is a starburst. *Hint: the true standardized slope is near $2.0 \cdot s_x/s_y$; check that your "sane" prior comfortably contains it.*

**3. (Conceptual — Gelman 2006.)** In your own words, give the two reasons `InverseGamma(ε, ε)` on a variance is a bad "noninformative" default in a hierarchical model, and explain *specifically* why the few-groups / small-true-$\sigma$ regime is where it bites hardest. What does Gelman recommend instead, and why does the half-Cauchy's heavy tail matter here? *Hint: think about what the prior does near $\sigma = 0$ and whether a sensible $\varepsilon \to 0$ limit exists.*

**4. (Coding — LKJ intuition.)** For `eta` in `{0.5, 1, 2, 10}`, build a model with `pm.LKJCholeskyCov("c", eta=eta, n=2, sd_dist=pm.Exponential.dist(1., size=2), compute_corr=True)` and *no* likelihood, sample the prior with `pm.sample_prior_predictive`, and plot the prior distribution of the off-diagonal correlation `c_corr[0,1]`. Confirm the direction of `eta` empirically: which value makes correlations cluster near 0, and which lets them roam toward ±1? *Hint: if your plot contradicts "larger `eta` ⇒ less correlation," re-read which way the density $\det(R)^{\eta-1}$ tilts.*

**5. (Coding — prior sensitivity.)** Take the Howell good model from §9 but fit it on a **random subsample of 12 adults**. Run the §8 sensitivity harness with slope-prior SDs `{0.5, 1, 5}` and a forest plot. Do the slope intervals overlap now, or separate? Write one sentence on what this tells you about when prior choice matters. *Hint: compare against the full-$n$ result — the contrast is the point.*

**6. (Conceptual + coding — GLM trap.)** Build a logistic regression with a single standardized predictor and coefficient prior `Normal(0, 10)`. Forward-simulate the prior predictive and plot the implied **probabilities** (`pm.math.invlogit(a + b*x)` as a `Deterministic`), *not* the log-odds. Describe what happens to the prior over probabilities, and explain why a coefficient prior that looked "mild" on the log-odds scale is actually extreme on the probability scale. *Hint: this previews Chapter 09 — the lesson is that "check on the outcome scale" is non-negotiable for GLMs. Access detail: wrap `p = pm.Deterministic("p", pm.math.invlogit(a + b*x))`; a `Deterministic` is stored in the **`prior`** group (not `prior_predictive`), so after `sample_prior_predictive` the draws live in `idata.prior["p"]` — histogram those (don't reach for `az.plot_ppc`). You should see a U-shape piling up at 0 and 1.*

---

## 📚 Resources & further reading

- **Stan, *Prior Choice Recommendations* (wiki)** — https://github.com/stan-dev/stan/wiki/prior-choice-recommendations — the single most concrete, opinionated source. The 5-level hierarchy (flat → super-vague → weak → generic weakly-informative `normal(0,1)`/`student_t(3,0,1)` → informative), the `normal(0, 2.5)` rstanarm default, the "Cauchy now discouraged" note, scale-parameter advice, and the `inv_gamma(0.4, 0.3)` NB-dispersion prior all come from here.
- **Gelman (2006), *Prior distributions for variance parameters in hierarchical models*** — https://sites.stat.columbia.edu/gelman/research/published/taumain.pdf — *Bayesian Analysis 1(3):515–534*. The canonical case against `InvGamma(ε,ε)` and for the half-Cauchy on the SD. Read this if you read nothing else on scale priors.
- **McElreath, *Statistical Rethinking* (2e, 2020), Chapter 4** — the prior-predictive-for-height exercise this chapter's Howell section mirrors. The PyMC port lives in `pymc-devs/pymc-resources` (Statistical Rethinking ports). The free lecture videos are superb on building prior intuition.
- **PyMC core notebook, *Prior and Posterior Predictive Checks*** — https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/posterior_predictive.html — the `standardize` / `sample_prior_predictive` idiom and the ±40-SD flat-prior cautionary tale; the pattern we copied in §2 and §9.
- **PyMC API — `sample_prior_predictive`** — https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.sample_prior_predictive.html — verified `draws=500` default and the `prior` / `prior_predictive` groups returned in InferenceData.
- **PyMC API — `LKJCholeskyCov`** — https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.LKJCholeskyCov.html — verified `eta`, `n`, `sd_dist`, `compute_corr` parameters and the runnable example; the reference for §5 (and Chapter 10).
- **Simpson, Rue, Riebler, Martins, Sørbye (2017), *Penalising model component complexity (PC priors)*** — https://arxiv.org/abs/1403.4630 — *Statistical Science 32(1):1–28*. The principled derivation behind "light-tailed, decreasing-from-zero priors on the SD," and the base-model way of thinking.
- **Carvalho, Polson & Scott (2010), *The horseshoe estimator for sparse signals*** — *Biometrika 97(2):465–480* (AISTATS precursor, 2009). The primary reference for the original horseshoe prior; pin this as the source of the global–local construction in §6.
- **Piironen & Vehtari (2017), *Sparsity information and regularization in the horseshoe…*** — https://arxiv.org/abs/1707.01694 — *EJS 11(2):5018–5051*. The regularized horseshoe: deriving $\tau_0$ from an expected number of nonzeros, and the finite-width slab. The reference for §6 (and Chapter 14).
- **Austin Rochford, *The Hierarchical Regularized Horseshoe Prior in PyMC*** — https://austinrochford.com/posts/2021-05-29-horseshoe-pymc3.html — the canonical worked PyMC build for the (regularized) horseshoe; porting to v5 is mechanical.
- **Kallioinen, Paananen, Bürkner, Vehtari (2023), *Detecting and diagnosing prior and likelihood sensitivity with power-scaling*** — https://arxiv.org/abs/2107.14054 — *Statistics & Computing 34:57*; package **priorsense** at https://n-kall.github.io/priorsense/. The modern automatic sensitivity method (R-only today; we do manual refits in §8).
- **Gelman et al. (2020), *Bayesian Workflow*** — https://arxiv.org/abs/2011.01808 — situates prior predictive checks and sensitivity inside the full iterative workflow (the backbone of Chapter 08).
- **Gabry, Simpson, Vehtari, Betancourt, Gelman (2019), *Visualization in Bayesian workflow*** — the visual grammar of prior/posterior predictive checks; pairs naturally with ArviZ.
- **Martin, Kumar, Lao, *Bayesian Modeling and Computation in Python*** — free at https://bayesiancomputationbook.com — the PyMC/ArviZ companion; see the chapters on the exploratory workflow and prior/posterior predictive checks.
- **BDA3 (Gelman et al. 2013), Chapters 2–5** — http://www.stat.columbia.edu/~gelman/book/ — conjugacy, noninformative vs. weakly-informative priors, and hierarchical variance priors; the reference text.
- **Bambi default-priors paper (Westfall 2017)** — https://arxiv.org/abs/1702.01201 — exactly how Bambi auto-scales weakly-informative priors (rstanarm-style); relevant when you let a formula API pick priors for you (Chapter 09). Inspect the generated priors with `model.build()` rather than hardcoding scale factors that drift across versions.

---

## ➡️ What's next

You now have the *judgment* half of Bayesian modeling: how to choose, build, simulate-check, and sensitivity-test priors so your model's small world is a sane one before any data arrive. But a model is only useful once you can compute its **posterior** — and we've been hand-waving `pm.sample` as a magic box. **Chapter 04 — Inference I: Analytic, Grid & Quadratic Approximation** opens that box from the bottom up. We'll compute posteriors exactly where conjugacy lets us (revisiting the Beta-Binomial and Normal-Normal updates with the priors you now know how to choose), approximate them on a grid to *see* the posterior surface, and meet the **quadratic / Laplace approximation** (McElreath's `quap`) — the Gaussian-at-the-mode shortcut that's fast, instructive, and the perfect stepping stone before we unleash MCMC and HMC/NUTS in Chapters 05–07. The priors you built here are the inputs to every one of those machines.
