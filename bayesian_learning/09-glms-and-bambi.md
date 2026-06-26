# Chapter 09 — Generalized Linear Models & Bambi

### One recipe for almost every regression you'll ever write — and the formula API that writes it for you

> *A note from me to you.*
>
> *Up to now you have built models one distribution at a time, by hand, thinking hard about each line. That is exactly the right way to learn. But here is a secret the textbooks bury in chapter twelve: the overwhelming majority of the regression models you will ever fit — linear, logistic, Poisson, negative binomial, robust, ordinal, multinomial — are the **same model**. They share one skeleton. Once you see the skeleton, you stop memorizing a zoo of unrelated techniques and start composing from three interchangeable parts: a **linear predictor**, a **link**, and a **likelihood family**. That is the Generalized Linear Model, and it is the most reusable idea in applied statistics.*
>
> *This chapter does two things. First, it teaches you the GLM recipe so deeply that you can derive any member of the family on a napkin and — more importantly — **interpret** it. Interpretation is where most practitioners quietly fail. They fit a logistic regression, read the coefficient as if it were a probability, and ship a wrong number to a stakeholder. They fit a Poisson model, forget that a city with ten times the population will have ten times the crimes for boring arithmetic reasons, and "discover" an effect that is pure exposure. I will drill the link-scale-vs-outcome-scale distinction into you until pushing predictions back to the human-readable scale is a reflex.*
>
> *Second, it introduces **Bambi** — the formula API that lets you write `bmb.Model("y ~ x", family="bernoulli")` and get a fully-specified, sanely-prior'd PyMC model for free. Bambi is a joy and a trap. A joy because it removes a hundred lines of boilerplate; a trap because it hides decisions you must still understand. So the rule of this chapter — the rule of this whole course — is: **every time I show you Bambi, I show you the raw PyMC model it generates.** You will never run a model you couldn't have written yourself. Let's build the skeleton.*

---

## What you'll be able to do after this chapter

- **State the GLM recipe** — linear predictor $\eta = \mathbf{X}\boldsymbol\beta$, link $g$, family $p(y\mid\mu)$ — and assemble any member of the family from these three knobs.
- **Fit linear, logistic, Poisson, negative-binomial, robust (Student-t), ordinal, and categorical regressions** in both raw PyMC v5 and Bambi, and explain exactly what Bambi generated under the hood.
- **Interpret coefficients on the right scale**: read logistic coefficients as **log-odds** and report **odds ratios** $e^{\beta}$; read Poisson coefficients as log-rates and report **incidence-rate ratios**; and *always* push predictions to the **outcome scale** for humans using the divide-by-4 rule and full posterior pushforwards.
- **Diagnose and cure (quasi-)separation** in logistic regression — recognize the runaway coefficient, and understand why a weakly-informative prior *is* the regularizer that fixes it.
- **Detect overdispersion** in count data with a posterior predictive check, and switch from Poisson to negative binomial — getting the PyMC `alpha`-vs-textbook-`phi` parametrization right (the single most common bug in this material).
- **Handle exposure with an offset** so you model rates, not raw counts.
- **Make a Normal regression fail on an outlier and a Student-t regression shrug it off**, and explain why.
- **Compute marginal effects, contrasts, and counterfactual predictions with full posterior uncertainty** using `bmb.interpret.predictions / comparisons / slopes`, and replicate the same quantities by hand in PyMC.
- **Read and write the Bambi formula grammar** — interactions (`a*b`), categorical coding (`C(x)`), success-level selection (`y['level']`), aggregated binomial (`p(s, n)`), and offsets.

I assume you are fluent with everything through **Chapter 08 (The Bayesian Workflow)**: you can write a `pm.Model`, set priors (we lean hard on the standardization-by-default habit from **Chapter 03**), run `pm.sample`, and read the diagnostics from **Chapter 07** ($\hat R \le 1.01$, ESS $\gtrsim 400$, zero divergences). Those gates apply to *every* model in this chapter; I will not re-derive them, I will just expect them to be green.

```python
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import bambi as bmb
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)
```

---

## 1. The GLM recipe: three knobs and nothing else

Let me give you the whole idea before any code, because everything else in this chapter is a special case of it.

A **Generalized Linear Model** is built from three parts, and *only* three:

1. **A linear predictor.** Take your predictors, multiply by coefficients, add them up:
   $$
   \eta_i \;=\; \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \cdots + \beta_p x_{ip} \;=\; \mathbf{x}_i^\top \boldsymbol\beta.
   $$
   This is the *only* part that is literally linear. It can range over all of $(-\infty, +\infty)$. In ASCII:
   ```
   eta_i = b0 + b1*x1 + b2*x2 + ... + bp*xp        # the linear predictor, a real number
   ```
   Everything you know about linear regression — adding predictors, interactions, polynomial terms, splines — is engineering on $\eta$. The "linear" in GLM refers to *linearity in the coefficients*, not in the data: $\beta_1 x^2$ is still a perfectly good linear-predictor term.

2. **A link function $g$.** The linear predictor lives on the whole real line, but the *mean* of your outcome often does not. A probability lives in $[0,1]$; an expected count lives in $[0,\infty)$. The link is a smooth, invertible function that connects them:
   $$
   g(\mu_i) = \eta_i \qquad\Longleftrightarrow\qquad \mu_i = g^{-1}(\eta_i).
   $$
   $g$ maps the constrained mean onto the unconstrained line; $g^{-1}$ (the **inverse link**, or **mean function**) maps the line back to where the mean is allowed to live. The link is the translator between the "model scale" (where we do linear algebra) and the "outcome scale" (where the data live).

3. **A likelihood family.** Finally, a probability distribution for the data given the mean — Normal, Bernoulli, Poisson, Gamma, and so on. This is what makes the model *generative*: it says how $y_i$ is actually produced once we know $\mu_i$.
   $$
   y_i \;\sim\; \text{Family}(\mu_i, \;\text{[extra parameters]}).
   $$

That's it. **Family, link, predictor.** Turn those three knobs and you get the entire catalog:

| Model | Family | Canonical link $g$ | Inverse link $g^{-1}$ | Outcome lives in |
|---|---|---|---|---|
| Linear regression | Normal | identity | identity | $\mathbb{R}$ |
| Logistic regression | Bernoulli / Binomial | logit | logistic / sigmoid | $\{0,1\}$ / $\{0,\dots,n\}$ |
| Poisson regression | Poisson | log | exp | $\{0,1,2,\dots\}$ |
| Negative-binomial | NegBinomial | log | exp | $\{0,1,2,\dots\}$ |
| Robust regression | Student-t | identity | identity | $\mathbb{R}$ |
| Gamma regression | Gamma | log (or inverse) | exp | $(0,\infty)$ |
| Ordinal regression | Categorical (ordered) | cumulative logit | — | ordered $\{1,\dots,K\}$ |
| Multinomial / softmax | Categorical | softmax | softmax | unordered $\{1,\dots,K\}$ |

> 💡 **Intuition:** A GLM is a Normal linear regression that has been taught two new tricks: (1) it can speak the native language of the outcome (counts, yes/no, categories) through the **link**, and (2) it can use the right noise distribution for that outcome through the **family**. The linear-algebra heart — "add up weighted predictors" — never changes. When you meet a new outcome type, you are not learning a new model; you are picking a new family and a new link and reusing everything else.

> 🧮 **The math (why a Normal regression *is* a GLM):** Ordinary linear regression is the GLM with family = Normal, link = identity. The "link" $g(\mu)=\mu$ does nothing because the mean of a Normal is already allowed to be any real number, so no translation is needed. Once you see that, linear regression stops being the special thing and becomes the *boring* corner of a much bigger room.

A crucial consequence, which I will repeat until you dream it: **coefficients live on the link scale, but humans live on the outcome scale.** In a Poisson model, $\beta_1 = 0.7$ does not mean "0.7 more events." It means the *log-rate* goes up by 0.7, i.e. the rate is multiplied by $e^{0.7}\approx 2.0$. The entire back half of this chapter is about crossing that bridge correctly, every time, with uncertainty attached.

> 📜 **Citation/Origin:** The GLM framework is Nelder & Wedderburn (1972), "Generalized Linear Models," *JRSS-A*. The Bayesian treatment we use — same skeleton, but with priors on $\boldsymbol\beta$ and full posterior interpretation — is the spine of McElreath's *Statistical Rethinking* Ch 10–12 and Gelman, Hill & Vehtari's *Regression and Other Stories* Ch 13–15. We are standing on fifty years of accumulated good taste.

### Why the Bayesian version is strictly better here

In frequentist GLMs you fit by maximum likelihood and lean on asymptotic normal approximations for standard errors. That works beautifully when data are plentiful and the likelihood is well-behaved — and it fails, sometimes catastrophically, exactly where it matters: small samples, rare events, **separation** (we'll see logistic coefficients run to $\pm\infty$), and any time you want a credible interval on a *transformed* quantity like a predicted probability or an odds ratio. The frequentist answer there is the delta method — a first-order Taylor approximation that is often wrong in the tails.

The Bayesian GLM fixes all of this with one mechanism: **priors regularize, and the posterior pushforward is exact.** A weakly-informative prior on $\boldsymbol\beta$ keeps coefficients finite under separation (we *prove* this in §3). And because we have posterior *draws*, we can push every draw through the inverse link and any downstream transformation, getting an exact posterior for "the probability of survival for a 30-year-old woman in first class" — no delta method, no approximation, just `g_inverse(eta_draws)`. Hold onto that idea; it is the engine behind `bmb.interpret` at the end of the chapter.

---

## 2. The baseline: linear regression, by hand and with Bambi

Let's anchor the recipe on the simplest member — the identity-link Normal GLM — using the **Howell !Kung census** data you first met in the early chapters. We model adult height from weight. This is deliberately a model you already know; the point is to watch the *same code shape* generalize.

```python
# Howell !Kung partial census (from McElreath's rethinking repo)
howell = pd.read_csv(
    "https://raw.githubusercontent.com/rmcelreath/rethinking/master/data/Howell1.csv",
    sep=";",
)
adults = howell[howell.age >= 18].copy()

# Standardize the predictor — our default for sane priors (Chapter 03).
adults["weight_z"] = (adults.weight - adults.weight.mean()) / adults.weight.std()
adults = adults.reset_index(drop=True)
print(adults[["height", "weight", "weight_z"]].describe().round(1))
```

> 🔧 **In practice:** We standardize `weight` to mean 0, SD 1. This is not cosmetic. After standardization the intercept is "expected height at the *average* weight" (a meaningful, well-identified quantity near 155 cm), and the slope is "centimeters per one-SD change in weight," which makes a `Normal(0, 10)` prior on the slope obviously reasonable. Standardization is the habit that lets you set priors without re-deriving the scale of your data every time. We established it in Chapter 03; from here on it is the default, not a special move.

### Raw PyMC

```python
coords = {"obs": adults.index}
with pm.Model(coords=coords) as howell_lin:
    weight = pm.Data("weight_z", adults.weight_z.values, dims="obs")   # mutable: enables counterfactuals

    a     = pm.Normal("Intercept", mu=155, sigma=20)   # avg adult height ~155cm, generous spread
    b     = pm.Normal("weight_z", mu=0, sigma=10)       # cm per SD of weight
    sigma = pm.HalfNormal("sigma", sigma=10)            # residual SD, positive

    mu = pm.Deterministic("mu", a + b * weight, dims="obs")           # eta == mu (identity link)
    pm.Normal("height", mu=mu, sigma=sigma, observed=adults.height.values, dims="obs")

    idata_lin = pm.sample(1000, tune=1000, chains=4, target_accept=0.9,
                          random_seed=RANDOM_SEED)
```

Notice the structure, because it never changes: declare data, put priors on coefficients and on the family's auxiliary parameter (`sigma`), build the **linear predictor**, apply the **inverse link** (here the identity, so `mu = eta`), and hand the result to the **family** (`pm.Normal`). Family, link, predictor — exactly as promised.

```python
az.summary(idata_lin, var_names=["Intercept", "weight_z", "sigma"], round_to=2)
```

You should see something like an intercept near $154.6$, a slope near $5.0$ (cm per SD of weight), and `sigma` near $5.1$ — all with $\hat R = 1.00$ and ESS in the thousands. Read it: *at average weight an adult !Kung is about 155 cm; moving one SD heavier (~6 kg) adds about 5 cm; individual heights scatter about $\pm 5$ cm around the line.* That sentence is the whole model in English.

### The same model in Bambi

Now watch Bambi compress all of that into one line:

```python
model_lin = bmb.Model("height ~ weight_z", data=adults, family="gaussian")
idata_lin_bmb = model_lin.fit(draws=1000, tune=1000, chains=4, target_accept=0.9,
                              random_seed=RANDOM_SEED)
az.summary(idata_lin_bmb, round_to=2)
```

That is the entire model. `family="gaussian"` selects the Normal likelihood with the identity link (the default for Gaussian); the intercept is implicit; `weight_z` becomes the slope. The returned object is an ordinary `az.InferenceData` — every ArviZ diagnostic you know works on it unchanged.

> 💡 **Intuition — what did Bambi just do for me?** It read the formula, found one predictor, and built *precisely* the PyMC model above — with one difference: it chose the priors for you, and it scales them to your data. `bmb.Model(..., auto_scale=True, center_predictors=True)` (both **on by default**) compute weakly-informative priors calibrated to the spread of `height` and `weight_z`. That is why a bare `bmb.Model("y ~ x", df)` already samples sanely instead of wandering. You did not turn off your brain; Bambi turned on a good default.

You should *never* trust a default you can't inspect. So inspect it:

```python
model_lin                       # printing the model lists the priors Bambi chose
# model_lin.build()             # build the underlying PyMC model without sampling
# model_lin.plot_priors()       # draw the prior densities
# model_lin.graph()             # render the model DAG (needs graphviz)
# model_lin.backend.model       # <-- the actual pm.Model object Bambi generated
```

Printing the model shows Bambi's auto-priors, which will look like `Intercept ~ Normal(mu=<mean of height>, sigma=<scaled>)` and `weight_z ~ Normal(mu=0, sigma=<scaled>)`. They are weakly-informative, not flat. If you want *your* priors instead of Bambi's, pass them explicitly:

```python
custom_priors = {
    "Intercept": bmb.Prior("Normal", mu=155, sigma=20),
    "weight_z":  bmb.Prior("Normal", mu=0, sigma=10),
    "sigma":     bmb.Prior("HalfNormal", sigma=10),
}
model_lin_custom = bmb.Model("height ~ weight_z", data=adults, family="gaussian",
                             priors=custom_priors)
```

Now the Bambi model and the raw-PyMC model are the *same model*, and they will return statistically indistinguishable posteriors. That round trip — write it in Bambi, reproduce it in PyMC, confirm they agree — is the discipline I want you to keep for every family below.

> ⚠️ **Pitfall:** A common confusion: "Bambi's default priors look too tight, my posterior barely moves from the prior." Almost always this means the data genuinely pin the parameter (good!), *or* you are comparing Bambi's standardized-scale prior against an intuition you formed on the raw, unstandardized scale. Bambi centers predictors internally, so its intercept prior is centered at the *outcome mean*, not at zero. Run `model.plot_priors()` and look, rather than guessing. We will hit this again with logistic models.

---

## 3. Logistic regression: the logit link, log-odds, and odds ratios

Now we turn the first new knob. The outcome is binary — survived or not, success or failure — so the family is **Bernoulli** and the natural quantity is a probability $p_i = P(y_i = 1)$. But a probability is trapped in $[0,1]$, and our linear predictor roams over all of $\mathbb{R}$. We need a link that maps $[0,1]$ onto the whole line. That link is the **logit**:

$$
\text{logit}(p) \;=\; \log\!\frac{p}{1-p} \;=\; \eta,
\qquad\qquad
p \;=\; \frac{1}{1+e^{-\eta}} \;=\; \sigma(\eta).
$$

The quantity $\frac{p}{1-p}$ is the **odds** — "how many times more likely the event is than its absence." Its logarithm, the **log-odds**, is what the linear predictor models. The inverse link $\sigma$ is the **logistic** (a.k.a. sigmoid) function, an S-curve squashing the line back into $(0,1)$. In ASCII:

```
logit(p) = log( p / (1-p) ) = eta        # link:   probability -> real line (log-odds)
p        = 1 / (1 + exp(-eta))           # inverse: real line -> probability (sigmoid)
```

> 💡 **Intuition:** The logit link exists because nature put probabilities on a leash. If we modeled $p$ directly with a line, the line would happily predict $p = 1.3$ or $p = -0.2$, which are nonsense. The logit stretches the leashed interval $[0,1]$ out to the full real line, lets the linear predictor run free there, and the sigmoid snaps it back. The S-shape is not arbitrary aesthetics — it is the geometric price of keeping probabilities legal.

### What a coefficient *means* in a logistic model

Here is the interpretation that practitioners get wrong constantly, so we go slowly. Suppose $\eta = \beta_0 + \beta_1 x$. Increase $x$ by one unit. The log-odds increase by exactly $\beta_1$. Exponentiate: the **odds** get *multiplied* by $e^{\beta_1}$. That multiplier is the **odds ratio**:

$$
\frac{\text{odds}(x+1)}{\text{odds}(x)} \;=\; \frac{e^{\beta_0 + \beta_1(x+1)}}{e^{\beta_0 + \beta_1 x}} \;=\; e^{\beta_1} \;=:\; \text{OR}.
$$

So: **logistic coefficients are log-odds; their exponentials are odds ratios.** $\beta_1 = 0.7 \Rightarrow \text{OR} = e^{0.7} \approx 2.0$: a one-unit increase in $x$ *doubles the odds*. $\beta_1 = 0 \Rightarrow \text{OR} = 1$: no effect. $\beta_1 < 0 \Rightarrow \text{OR} < 1$: the odds shrink.

What a coefficient is **not** is a change in probability. The probability change for a one-unit move in $x$ depends on *where you start*: near $p = 0.5$ the sigmoid is steep and a coefficient moves probability a lot; out near $p = 0.02$ or $p = 0.98$ the sigmoid is flat and the *same* coefficient barely moves probability at all. This is the number-one logistic-regression error: reading a log-odds coefficient as if it were a probability.

> 🧮 **The math — the divide-by-4 rule.** How much can a coefficient move probability *at most*? The slope of $p = \sigma(\eta)$ with respect to $x$ is $\beta_1\, \sigma(\eta)\,(1-\sigma(\eta))$. The term $\sigma(1-\sigma)$ is maximized at $p = 0.5$, where it equals $1/4$. So the steepest the probability curve ever gets, per unit $x$, is $\beta_1 / 4$. That gives Gelman & Hill's famous shortcut: **divide the coefficient by 4 to get an upper bound on the probability change** near the middle. A coefficient of $0.7$ moves probability by at most $\approx 0.175$ per unit — and *less* than that everywhere except right at $p=0.5$. Memorize this; it lets you sanity-check a logistic coefficient in your head.

> 📜 **Citation/Origin:** The divide-by-4 rule and the "always interpret on the probability scale" discipline are from Gelman, Hill & Vehtari, *Regression and Other Stories*, Ch 13. The odds-ratio framing traces to epidemiology, where it became standard because odds ratios are invariant to certain study designs.

### The dataset: who survived the Titanic?

```python
titanic = sns.load_dataset("titanic")          # survived, pclass, sex, age, fare, ...
df = titanic.dropna(subset=["age"]).copy()     # age has genuine NaNs — Ch 16 models them; here we drop
df["female"]   = (df["sex"] == "female").astype(int)
df["age_z"]    = (df.age  - df.age.mean())  / df.age.std()
df["fare_z"]   = (df.fare - df.fare.mean()) / df.fare.std()
df = df.reset_index(drop=True)
print(df.groupby("sex").survived.mean().round(2))   # the "women and children first" signal
```

### Raw PyMC — Bernoulli with a logit link

```python
coords = {"obs": df.index}
with pm.Model(coords=coords) as titanic_logit:
    female = pm.Data("female", df.female.values, dims="obs")
    age_z  = pm.Data("age_z",  df.age_z.values,  dims="obs")
    fare_z = pm.Data("fare_z", df.fare_z.values, dims="obs")

    # Weakly-informative priors on the LOG-ODDS scale (Gelman 2008). On standardized
    # predictors, Normal(0, 1.5) says "I doubt any single SD move shifts the odds by
    # more than ~e^3 = 20x" — generous but not insane.
    b0   = pm.Normal("Intercept", mu=0, sigma=1.5)
    bsex = pm.Normal("female",    mu=0, sigma=1.5)
    bage = pm.Normal("age_z",     mu=0, sigma=1.5)
    bfar = pm.Normal("fare_z",    mu=0, sigma=1.5)

    eta = b0 + bsex * female + bage * age_z + bfar * fare_z
    p   = pm.Deterministic("p", pm.math.invlogit(eta), dims="obs")     # inverse link = sigmoid

    # Pass logit_p directly — numerically safer than building p then handing it over.
    pm.Bernoulli("survived", logit_p=eta, observed=df.survived.values, dims="obs")

    idata_logit = pm.sample(1000, tune=1000, chains=4, target_accept=0.9,
                            random_seed=RANDOM_SEED)
```

> 🔧 **In practice:** `pm.Bernoulli` accepts `logit_p=` directly. Prefer it over computing `p = invlogit(eta)` and passing `p=`: the `logit_p` form keeps the computation on the log-odds scale internally and is numerically stable when $\eta$ is large in magnitude (where `invlogit` would round to exactly 0 or 1 and the log-likelihood would underflow to $-\infty$). I still build a `Deterministic("p", ...)` separately so I have probabilities saved for plotting — but the *likelihood* uses `logit_p`.

```python
az.summary(idata_logit, var_names=["Intercept", "female", "age_z", "fare_z"], round_to=2)
```

Now the interpretation drill. Suppose `female` comes back at $2.5$. On the **log-odds** scale that is large. Exponentiate for the **odds ratio**: $e^{2.5} \approx 12$ — *being female multiplies the odds of survival by about twelve.* Apply divide-by-4 for a probability feel: $2.5/4 \approx 0.62$, so near the middle of the curve being female lifts survival probability by up to ~62 percentage points. Both numbers are correct; the odds ratio is the precise multiplicative statement, the divide-by-4 is the quick probability gut-check. **Report odds ratios to statisticians and probability differences to everyone else** — and we'll compute the latter exactly, with uncertainty, in §10.

```python
# Posterior of the odds ratios — exact, no delta method.
post = az.extract(idata_logit)
for name in ["female", "age_z", "fare_z"]:
    or_draws = np.exp(post[name].values)
    lo, hi = az.hdi(or_draws, hdi_prob=0.94)   # real HDI, not a [3,97] percentile interval
    print(f"OR[{name}] = {or_draws.mean():.2f}  (94% HDI [{lo:.2f}, {hi:.2f}])")
```

Because we have draws, the odds-ratio posterior is just `exp` of the coefficient draws — a genuine posterior distribution, not a point estimate with an asymptotic standard error. That is the Bayesian pushforward, and it is exact.

### The same model in Bambi

```python
model_logit = bmb.Model(
    "survived ~ female + age_z + fare_z",
    data=df, family="bernoulli",          # logit link is the default for bernoulli
)
idata_logit_bmb = model_logit.fit(
    draws=1000, tune=1000, chains=4, target_accept=0.9, random_seed=RANDOM_SEED,
    idata_kwargs={"log_likelihood": True},     # store pointwise log-lik for az.loo / az.compare
)
az.summary(idata_logit_bmb, round_to=2)
```

`family="bernoulli"` picks the Bernoulli likelihood and — automatically — the logit link. The `idata_kwargs={"log_likelihood": True}` is the one piece of bookkeeping you must not forget when you intend to do model comparison later; it tells PyMC to store the pointwise log-likelihood that `az.loo` and `az.compare` need (Chapter 11).

> ⚠️ **Pitfall — the string-outcome footgun.** If your outcome is a string or has more than two levels, Bambi cannot guess which level is the "success." It will raise "could not determine success level." Tell it explicitly with the bracket syntax: `bmb.Model("vote['clinton'] ~ party_id", ...)` marks `'clinton'` as the event. For our 0/1 `survived` column this is unnecessary, but you will hit it the moment you model a categorical string outcome.

### Quasi-separation: when a flat prior lets a coefficient run to infinity

This is the headline reason logistic regression *needs* Bayes, so we are going to make it fail and then cure it.

**Separation** happens when a predictor (or combination) perfectly — or almost perfectly — predicts the outcome. Imagine a feature that is 1 for *every* survivor and 0 for *every* casualty. The maximum-likelihood fit wants to push that coefficient to $+\infty$, because the larger it gets, the closer the predicted probabilities march to a perfect 0/1 split, and the likelihood keeps creeping up *forever* with no finite maximum. Frequentist software either fails to converge, reports an astronomical coefficient with an even more astronomical standard error, or silently returns garbage. With a flat/very-wide prior, a Bayesian sampler inherits the same disease: the posterior has no information to stop the coefficient, so it wanders toward infinity, $\hat R$ explodes, and you get divergences.

Let me construct it deliberately:

```python
# A toy separation: a binary feature that perfectly predicts the outcome.
n = 60
x_sep = np.r_[np.zeros(n // 2), np.ones(n // 2)]        # first half 0, second half 1
y_sep = x_sep.copy().astype(int)                        # outcome == feature: PERFECT separation

# (a) Near-flat prior -> the coefficient tries to run to +infinity.
with pm.Model() as sep_flat:
    b0 = pm.Normal("Intercept", 0, 100)                 # essentially flat
    b1 = pm.Normal("x", 0, 100)                         # essentially flat
    pm.Bernoulli("y", logit_p=b0 + b1 * x_sep, observed=y_sep)
    idata_sep_flat = pm.sample(1000, tune=1000, chains=4, target_accept=0.95,
                               random_seed=RANDOM_SEED)
```

When you run this you will see the symptoms from Chapter 07 light up: the summary for `x` shows an enormous posterior mean (tens or hundreds), an enormous SD, $\hat R$ well above $1.01$, tiny ESS, and a pile of divergences. The chain is climbing a likelihood ridge that never crests.

Now the cure — and it is *only* the prior:

```python
# (b) Weakly-informative prior -> the coefficient is regularized to a finite value.
with pm.Model() as sep_reg:
    b0 = pm.Normal("Intercept", 0, 1.5)
    b1 = pm.StudentT("x", nu=4, mu=0, sigma=1.0)        # Gelman et al. (2008) t(4,0,1) variant
    pm.Bernoulli("y", logit_p=b0 + b1 * x_sep, observed=y_sep)
    idata_sep_reg = pm.sample(1000, tune=1000, chains=4, target_accept=0.95,
                              random_seed=RANDOM_SEED)
az.summary(idata_sep_reg, var_names=["Intercept", "x"], round_to=2)
```

Now `x` settles at a large-but-finite value (a few units), $\hat R = 1.00$, ESS healthy, divergences gone. The prior supplied the missing information: "coefficients this size are implausible," which is exactly the regularization the data could not provide on their own. The data *still* dominate the prior wherever they carry information; the prior only acts where the likelihood is flat — i.e. precisely on the runaway direction.

> 💡 **Intuition — the prior IS the regularizer.** Under separation the likelihood is a ramp with no peak. Multiply a ramp by a bump (the prior) and you get a peak — a proper, finite posterior. This is the same mathematics as L2 / ridge regularization in machine learning (a Gaussian prior is *literally* an L2 penalty on the coefficients), but here it falls out of Bayes' theorem with no extra machinery, and it comes with honest uncertainty rather than a single penalized point estimate.

> 📜 **Citation/Origin:** Gelman, Jakulin, Pittau & Su (2008), "A weakly informative default prior distribution for logistic and other regression models," recommend a heavy-tailed-but-proper prior on coefficients of *standardized* (rescaled) predictors. Their headline default is a **Cauchy(0, 2.5)** — i.e. Student-t with $\nu=1$, scale $2.5$ — with a lighter-tailed **Student-t(ν=4, 0, 1)** offered as an alternative; we use the t(4, 0, 1) variant above. Either way the heavy tail lets a genuinely large true effect through while still preventing the infinite runaway. Bambi's defaults are built in this spirit — which is why a bare Bambi logistic model does *not* blow up on separated data. arXiv:0901.4011.

> 🩺 **Diagnostic:** Whenever a logistic (or any GLM) coefficient shows a huge posterior mean *and* huge SD *and* bad $\hat R$ together, suspect separation before you suspect a bug. Check `pd.crosstab(predictor, outcome)` for a row or column of zeros — the fingerprint of a predictor that perfectly splits the classes. The fix is never "more iterations"; it is a sensible prior.

### Aggregated binomial: the same model, counted

Sometimes data arrive pre-aggregated: out of $n$ passengers in a cabin class, $s$ survived. That is a **Binomial** GLM — same logit link, same coefficients, just counted trials instead of individual rows. In raw PyMC:

```python
# successes out of trials, e.g. survivors per (sex x class) cell
pm.Binomial("survived", n=trials, logit_p=eta, observed=successes, dims="cell")
```

and in Bambi the proportion helper `p(successes, trials)` is the confirmed current syntax:

```python
# pseudocode shape — group df first into successes/trials per cell
agg_model = bmb.Model("p(survived, total) ~ sex + pclass",
                      data=agg_df, family="binomial")
```

Bernoulli and aggregated Binomial are the *same* GLM — a Bernoulli model is just the Binomial with $n=1$ on every row. Whether you carry individual rows or cell counts is a data-layout choice, not a modeling one. (Chapter 10 leans on this `p(y, n)` syntax for hierarchical binomial models, so keep it in mind.)

---

## 4. Poisson regression: the log link, rates, and incidence-rate ratios

Counts — bike rentals per hour, disease cases per county, defects per batch — are non-negative integers with no upper bound. The natural family is **Poisson**, whose single parameter $\mu$ is simultaneously its mean *and* its variance (remember that; it is the whole story of the next section). The expected count $\mu$ must be positive, so the link that maps $(0,\infty)$ onto the line is the **log**:

$$
\log(\mu) = \eta, \qquad\qquad \mu = e^{\eta}.
$$

```
log(mu) = eta            # link:   positive rate -> real line
mu      = exp(eta)       # inverse: real line -> positive rate
```

The log link has a gorgeous consequence: because the inverse is exponential, **additive changes on the link scale become multiplicative changes on the count scale.** Increase $x$ by one unit and $\eta$ rises by $\beta_1$; the rate is multiplied by $e^{\beta_1}$. That multiplier is the **incidence-rate ratio (IRR)** — the count analogue of the odds ratio:

$$
\frac{\mu(x+1)}{\mu(x)} = e^{\beta_1} =: \text{IRR}.
$$

So $\beta_1 = 0.3 \Rightarrow \text{IRR} = e^{0.3}\approx 1.35$: a one-unit increase multiplies the expected count by 1.35, i.e. **+35%**. $\beta_1 = 0 \Rightarrow$ IRR $=1$, no effect. $\beta_1 < 0$ shrinks the count. As always, the coefficient is *not* a count change; it is a log-rate change, and $e^{\beta}$ is the human-readable multiplier.

> 💡 **Intuition:** The log link is the right choice almost any time effects feel *proportional* rather than *additive*. "Rain cuts ridership roughly in half" is multiplicative — it is an IRR of ~0.5 — and that is true whether the baseline is 20 riders or 2000. Modeling counts on the log scale bakes in this proportional-effects assumption, which is usually what you actually believe about the world.

### The dataset: hourly bike rentals

```python
bikes = bmb.load_data("bikes")        # registered Bambi dataset: hourly counts + weather
print(bikes.columns.tolist())
print(bikes[["count", "temperature", "humidity", "hour"]].describe().round(1))
```

The `bikes` data record the number of bikes rented per hour along with temperature, humidity, and the hour of day. Let's model `count` from `temperature` (standardized) — we expect more rentals when it's warmer, up to a point.

### Raw PyMC — Poisson with a log link

```python
bikes = bikes.reset_index(drop=True)
bikes["temp_z"] = (bikes.temperature - bikes.temperature.mean()) / bikes.temperature.std()

coords = {"obs": bikes.index}
with pm.Model(coords=coords) as bike_pois:
    temp = pm.Data("temp_z", bikes.temp_z.values, dims="obs")

    # Priors on the LOG scale. A Normal(0,1) on a log-rate slope is already a big effect:
    # e^1 = 2.7x per SD. Keep log-scale priors TIGHT or exp() will explode.
    b0 = pm.Normal("Intercept", mu=np.log(bikes["count"].mean()), sigma=1)
    b1 = pm.Normal("temp_z", mu=0, sigma=1)

    mu = pm.Deterministic("mu", pm.math.exp(b0 + b1 * temp), dims="obs")    # inverse link = exp
    pm.Poisson("count", mu=mu, observed=bikes["count"].values, dims="obs")

    idata_pois = pm.sample(1000, tune=1000, chains=4, target_accept=0.9,
                           random_seed=RANDOM_SEED)
az.summary(idata_pois, var_names=["Intercept", "temp_z"], round_to=2)
```

> ⚠️ **Pitfall — `exp` overflows.** The single most common Poisson bug is a runaway linear predictor: if priors on the log scale are too wide, the sampler proposes a large $\eta$, `exp(eta)` overflows to `inf`, the log-probability becomes `nan`, and sampling gets stuck. Two defenses, both of which we used: **(1) standardize predictors**, so a unit move is one SD, not one raw degree; **(2) keep log-scale priors tight** — `Normal(0,1)` on a slope is *already* a 2.7× rate change per SD, which is enormous. Anchoring the intercept prior at `log(mean count)` also starts the sampler in the right neighborhood. Exponentiate exactly once, inside the model.

Interpret it. If `temp_z` comes back at $0.4$, the IRR is $e^{0.4}\approx 1.49$ — *each one-SD increase in temperature is associated with about 49% more rentals.* The intercept, exponentiated, is the expected count at average temperature. Report IRRs, never raw log-coefficients, to anyone who has to make a decision.

### The same model in Bambi

```python
model_pois = bmb.Model("count ~ temp_z", data=bikes, family="poisson")  # log link default
idata_pois_bmb = model_pois.fit(draws=1000, tune=1000, chains=4, target_accept=0.9,
                                random_seed=RANDOM_SEED,
                                idata_kwargs={"log_likelihood": True})
az.summary(idata_pois_bmb, round_to=2)
```

`family="poisson"` selects the Poisson likelihood and the log link automatically. Same three knobs, one line.

### Exposure and the offset: model rates, not raw counts

Here is a trap that has produced more bogus "findings" than almost anything in applied statistics. Suppose you count crimes per city. A city ten times larger will have roughly ten times the crimes — for the boring arithmetic reason that there are ten times as many people, *not* because it is more dangerous. If you regress raw counts on city features, population swamps everything and you "discover" effects that are pure **exposure**.

The fix is to model the **rate** (events per unit of exposure) rather than the raw count. Let $u_i$ be the exposure of unit $i$ — population, person-years, road-kilometers, hours observed, area surveyed. We want $\mu_i = u_i \cdot \lambda_i$ where $\lambda_i$ is the rate we actually care about. Take logs:

$$
\log \mu_i = \log u_i + \underbrace{\mathbf{x}_i^\top\boldsymbol\beta}_{\log \lambda_i}.
$$

The term $\log u_i$ enters the linear predictor with its coefficient **pinned to 1** — it is *not* a free parameter to estimate. A predictor with a fixed coefficient of 1 is called an **offset**. It says "I know exactly how exposure scales the count (linearly); don't waste a parameter estimating it; just account for it."

In raw PyMC, the offset is simply a known column added to $\eta$:

```python
# pseudocode shape — suppose `exposure` is hours-of-operation per observation
log_exposure = np.log(bikes["exposure"].values)     # a known constant vector
with pm.Model(coords=coords) as bike_offset:
    temp = pm.Data("temp_z", bikes.temp_z.values, dims="obs")
    off  = pm.Data("log_exposure", log_exposure, dims="obs")
    b0 = pm.Normal("Intercept", 0, 1)
    b1 = pm.Normal("temp_z", 0, 1)
    mu = pm.math.exp(b0 + b1 * temp + off)           # offset added on the LOG scale, coef pinned to 1
    pm.Poisson("count", mu=mu, observed=bikes["count"].values, dims="obs")
```

In Bambi the offset goes right in the formula:

```python
# pseudocode — verify offset() imports/runs in your pinned Bambi; else use the raw-PyMC offset above
model_off = bmb.Model("count ~ temp_z + offset(log(exposure))",
                      data=bikes, family="poisson")
```

> 🔧 **In practice:** The Bambi `offset(log(exposure))` term is confirmed by the maintainer and present in the test suite, but he flagged it as lightly documented — so smoke-test it on your pinned version. If it errors, fall back to the raw-PyMC offset (a known `log_exposure` column added to $\eta$), which always works. Either way the modeling content is identical: the offset is exposure entering on the log scale with a coefficient of one.

> ⚠️ **Pitfall:** Forgetting the offset and then interpreting the result as if it were a rate. If your `count` plausibly scales with some "size" variable (population, time, area) and you do not offset, your coefficients are contaminated by exposure. The tell is a coefficient on a size-correlated predictor that is implausibly large and "explains" most of the variance — it is laundering population through the back door.

---

## 5. Overdispersion → Negative Binomial: when Poisson's variance is a lie

The Poisson model carries a rigid, load-bearing assumption that is easy to forget and almost always violated in real data:

$$
\mathrm{Var}(y_i) = \mathbb{E}(y_i) = \mu_i \qquad\text{(equidispersion).}
$$

Poisson has *one* knob, $\mu$, and it sets the mean and the variance simultaneously. Real count data are almost always **overdispersed** — more variable than their mean — because of unmodeled heterogeneity: bike rentals vary hour-to-hour for reasons beyond temperature; disease counts cluster; defects come in bad batches. When the true variance exceeds the mean, a Poisson model is *forced* to compromise: it cannot widen its predictive spread without also inflating its mean, so it produces predictive intervals that are far too narrow. Your model will look confident and be wrong.

### Detecting overdispersion with a posterior predictive check

You do not diagnose overdispersion by staring at coefficients; you diagnose it by asking the fitted model to **generate fake data** and checking whether the fake data are as variable as the real data. This is the posterior predictive check from Chapter 08, now doing real work.

```python
with bike_pois:                       # the Poisson model from §4
    pm.sample_posterior_predictive(idata_pois, extend_inferencedata=True,
                                   random_seed=RANDOM_SEED)

az.plot_ppc(idata_pois, num_pp_samples=100)      # overlay simulated vs observed count distribution
```

> 🩺 **Diagnostic — how to read this PPC.** `az.plot_ppc` overlays many simulated count distributions (thin light lines) on the observed distribution (thick dark line). For an *overdispersed* dataset under a Poisson fit you will see the classic signature: the observed distribution has a **fatter right tail and more mass at zero** than any of the simulated ones; the simulated curves are all bunched too tightly around the mean. The Poisson literally cannot manufacture the spread it needs. A complementary, sharper check is to compare a **test statistic** — the observed *variance-to-mean ratio* should sit near 1 under a true Poisson; if the observed ratio is 3, 5, 10, your data are 3–10× overdispersed:

```python
# Variance-to-mean ratio: ~1 for true Poisson, >>1 signals overdispersion.
obs = bikes["count"].values
print("observed Var/Mean =", round(obs.var() / obs.mean(), 2))

# Posterior-predictive distribution of the same statistic (should bracket 1 if Poisson is right)
ppc = idata_pois.posterior_predictive["count"].stack(s=("chain", "draw")).values  # (obs, draws)
ratio_rep = ppc.var(axis=0) / ppc.mean(axis=0)
print("PPC Var/Mean 94% interval =", np.round(np.percentile(ratio_rep, [3, 97]), 2))
```

If the observed ratio (say, $4.2$) sits far outside the model's predictive interval for that ratio (say, $[0.9, 1.1]$), the model is decisively rejected for under-representing variance. That is your cue to reach for the negative binomial.

### The Negative Binomial: Poisson with a variance dial

The **Negative Binomial** adds a second parameter that lets variance exceed the mean. PyMC parametrizes it by `(mu, alpha)` with

$$
\mathbb{E}(y) = \mu, \qquad \mathrm{Var}(y) = \mu + \frac{\mu^2}{\alpha}.
$$

The extra term $\mu^2/\alpha$ is the overdispersion. Read the limits carefully because they are counterintuitive:

- **Small $\alpha$ ⇒ large $\mu^2/\alpha$ ⇒ heavy overdispersion.**
- **$\alpha \to \infty$ ⇒ the extra term vanishes ⇒ the model collapses back to Poisson.**

So in PyMC, *small alpha means more overdispersion*. This is exactly backwards from how most textbooks write it, and it causes a genuinely enormous amount of confusion, so we slow down.

> ⚠️ **Pitfall — the α-vs-φ inversion (read this twice).** Many textbooks write the NB2 variance as $\mathrm{Var}(y) = \mu + \phi\,\mu^2$, where the dispersion parameter $\phi$ is in the *numerator*: **big $\phi$ = more overdispersion, $\phi \to 0$ = Poisson.** PyMC's `alpha` is the *reciprocal*: $\alpha_{\text{PyMC}} = 1/\phi$. So a PyMC `alpha` of 1000 is *near-Poisson*, while an `alpha` of 0.5 is *wildly overdispersed*. If you read PyMC's `alpha` as if it were the textbook $\phi$ you will get the dispersion story exactly inverted — and conclude "no overdispersion" precisely when there is a ton. Burn this in: **in PyMC, small `alpha` = big variance.** This is the single most common interpretation bug in this entire chapter.

> 🧮 **The math — where NB comes from.** The negative binomial is a Poisson whose rate is itself random — a **Gamma-Poisson mixture**: $y \mid \lambda \sim \text{Poisson}(\lambda)$, $\lambda \sim \text{Gamma}(\alpha, \alpha/\mu)$. Marginalizing out the per-observation rate $\lambda$ gives the NB. That is the literal generative story of overdispersion: each observation has its *own* unobserved rate drawn around $\mu$, and that extra layer of rate-variability is the surplus variance. It is no coincidence that this looks like a baby hierarchical model — it is one, with the group effect integrated out.

### Raw PyMC — Negative Binomial

```python
coords = {"obs": bikes.index}
with pm.Model(coords=coords) as bike_nb:
    temp = pm.Data("temp_z", bikes.temp_z.values, dims="obs")

    b0 = pm.Normal("Intercept", mu=np.log(bikes["count"].mean()), sigma=1)
    b1 = pm.Normal("temp_z", mu=0, sigma=1)
    alpha = pm.Exponential("alpha", 1.0)          # dispersion; SMALL alpha = MORE overdispersed

    mu = pm.Deterministic("mu", pm.math.exp(b0 + b1 * temp), dims="obs")
    pm.NegativeBinomial("count", mu=mu, alpha=alpha,
                        observed=bikes["count"].values, dims="obs")

    idata_nb = pm.sample(1000, tune=1000, chains=4, target_accept=0.9,
                         random_seed=RANDOM_SEED)

with bike_nb:
    pm.sample_posterior_predictive(idata_nb, extend_inferencedata=True,
                                   random_seed=RANDOM_SEED)
az.plot_ppc(idata_nb, num_pp_samples=100)
```

Re-run the PPC and the variance-to-mean check on `idata_nb`. Now the simulated distributions should *cover* the observed fat tail, and the predictive variance-to-mean interval should bracket the observed value. The `alpha` posterior tells you how much overdispersion the data demanded: a posterior mean of, say, $1.5$ implies substantial extra variance (recall $\mathrm{Var} = \mu + \mu^2/1.5$), whereas an `alpha` in the hundreds would have said "Poisson was fine after all."

### The same model in Bambi, and confirming with LOO

```python
model_nb = bmb.Model("count ~ temp_z", data=bikes, family="negativebinomial")  # log link default
idata_nb_bmb = model_nb.fit(draws=1000, tune=1000, chains=4, target_accept=0.9,
                            random_seed=RANDOM_SEED,
                            idata_kwargs={"log_likelihood": True})

# Formal comparison — does NB actually predict better out of sample? (Chapter 11)
cmp = az.compare({"poisson": idata_pois_bmb, "negbinomial": idata_nb_bmb})
print(cmp)
```

`az.compare` ranks the models by expected log predictive density (ELPD) via PSIS-LOO. On overdispersed data the negative binomial wins decisively — a large ELPD difference relative to its standard error — because it stops being absurdly overconfident about the variance. That is the full loop: a PPC *revealed* the misfit, the NB family *fixed* it, and LOO *confirmed* the fix predicts better. (We unpack `az.compare`, ELPD, and the $\hat k$ diagnostic properly in Chapter 11; here just note that the chosen family changed the answer.)

> 🔧 **In practice — when to also check for zero-inflation.** If even the negative binomial undershoots the *number of zeros* in your data (the PPC shows too little mass at exactly 0 while the rest of the fit is fine), you likely have a **zero-inflation** or **hurdle** process — two distinct generators, one producing structural zeros. The fix is `pm.ZeroInflatedPoisson` / `pm.ZeroInflatedNegativeBinomial`, which we cover in Chapter 13 (Mixture Models). Negative binomial handles *general* overdispersion; it does not handle a separate "always zero" subpopulation.

---

## 6. Robust regression: a Student-t likelihood that shrugs off outliers

Back to continuous outcomes — but with a problem the Normal family cannot handle. The Normal likelihood has **thin tails**: it assigns vanishingly small probability to points far from the mean. So when a single outlier lands far away, the Normal model faces a stark choice — either declare that point astronomically improbable (huge penalty) or *move the entire regression line toward it* to reduce the penalty. It chooses the latter. One bad point can swing a slope, because under a Normal likelihood every point, no matter how extreme, pulls with full force.

The cure is to swap the family for one with **heavy tails**: the **Student-t**. Same identity link, same linear predictor, but the t-distribution's polynomial (rather than exponential) tail decay means a far-away point is "merely unusual," not "impossible." The model can leave an outlier sitting out in the tail without paying a catastrophic likelihood penalty — so it *doesn't* distort the line to chase it. The degrees-of-freedom parameter $\nu$ tunes the tail weight:

- Small $\nu$ (say 1–7): heavy tails, very robust to outliers.
- Large $\nu$ ($\to\infty$): the Student-t *converges to the Normal*. So the Normal model is the $\nu=\infty$ corner of the robust model — robust regression strictly generalizes it.

> 💡 **Intuition:** Think of $\nu$ as a *skepticism dial*. A small $\nu$ says "I expect occasional wild observations; don't let them dominate." A large $\nu$ says "I trust that every point is informative; treat them all as Normal." The beautiful part is that you can *let the data choose $\nu$*. If there are real outliers, the posterior for $\nu$ concentrates low and the line ignores them. If the data are clean, $\nu$ wanders high and the model gracefully reduces to ordinary regression. You get robustness for free, only when you need it.

### Make the Normal fail, watch the Student-t hold

Let me generate clean linear data, fit a Normal model, then drop in a single gross outlier and refit both a Normal and a Student-t. The contrast is the lesson.

```python
# Clean linear data with known truth: y = 2 + 3*x + noise
N = 50
x = np.linspace(-2, 2, N)
y = 2.0 + 3.0 * x + rng.normal(0, 1.0, size=N)

# Corrupt ONE point: move it wildly off the line.
y_out = y.copy()
y_out[N // 2] += 25.0          # a single, gross outlier

def fit_likelihood(xv, yv, robust: bool):
    with pm.Model() as m:
        a = pm.Normal("Intercept", 0, 10)
        b = pm.Normal("slope", 0, 10)
        sigma = pm.HalfNormal("sigma", 5)
        mu = a + b * xv
        if robust:
            nu = pm.Exponential("nu", 1 / 29.0) + 1     # Stan/McElreath convention; nu > 1
            pm.StudentT("y", nu=nu, mu=mu, sigma=sigma, observed=yv)
        else:
            pm.Normal("y", mu=mu, sigma=sigma, observed=yv)
        idata = pm.sample(1000, tune=1000, chains=4, target_accept=0.9,
                          random_seed=RANDOM_SEED, progressbar=False)
    return idata

idata_normal_out = fit_likelihood(x, y_out, robust=False)
idata_robust_out = fit_likelihood(x, y_out, robust=True)

print("Normal  slope:", az.summary(idata_normal_out, var_names=["slope"])["mean"].item())
print("StudentT slope:", az.summary(idata_robust_out, var_names=["slope"])["mean"].item())
```

The true slope is $3.0$. With the outlier present, the **Normal** model's slope estimate is dragged noticeably away from 3 (and its `sigma` inflates as the model tries to "explain" the outlier by claiming the noise is huge). The **Student-t** model's slope stays close to 3, and its `nu` posterior concentrates at a *small* value — the model has detected the heavy tail and downweighted the rogue point. Plot both fitted lines over the scatter and the picture is unmistakable: the Normal line tilts toward the outlier; the robust line ignores it and tracks the bulk of the data.

> 🩺 **Diagnostic — read $\nu$ as an outlier detector.** After fitting a Student-t model, look at the posterior of $\nu$. A posterior concentrated at small values ($\nu \lesssim 10$) is the model *telling you* it found heavy tails / outliers and is actively protecting you. A posterior of $\nu$ that drifts to large values means the data are effectively Normal and robustness wasn't needed — which is itself a useful, reassuring finding. Either way, $\nu$ is a free diagnostic of how outlier-prone your data are.

> 🧮 **The math — why heavy tails resist outliers.** A Normal log-likelihood penalizes a residual $r$ quadratically: $-\tfrac{r^2}{2\sigma^2}$. Double the residual, quadruple the penalty — so the optimizer will pay almost anything to shrink a large residual, including tilting the whole line. The Student-t log-likelihood grows only *logarithmically* for large $r$: $-\tfrac{\nu+1}{2}\log(1 + \tfrac{r^2}{\nu\sigma^2})$. Past a point, making an outlier's residual even larger barely changes the penalty — so the model stops caring and leaves it in the tail. That bounded influence is *exactly* robustness, and it falls straight out of the tail shape.

### The same model in Bambi

```python
df_rob = pd.DataFrame({"x": x, "y": y_out})

model_normal = bmb.Model("y ~ x", data=df_rob, family="gaussian")
model_robust = bmb.Model("y ~ x", data=df_rob, family="t")      # Student-t likelihood

idata_n = model_normal.fit(draws=1000, tune=1000, chains=4, random_seed=RANDOM_SEED)
idata_t = model_robust.fit(draws=1000, tune=1000, chains=4, random_seed=RANDOM_SEED)
```

`family="t"` is the entire change. Bambi puts a sensible prior on the degrees-of-freedom parameter and handles the rest. Same one-knob swap as everywhere else in this chapter — that is the payoff of seeing every model as a GLM.

> 🔧 **In practice:** Make `family="t"` your default for *any* regression where you suspect contamination, heavy-tailed noise, or data-entry errors — which is to say, most real datasets. The cost is one extra parameter and a slightly slower fit; the benefit is that a single fat-fingered value can't silently corrupt your conclusions. When in doubt, fit both Normal and Student-t and compare with `az.compare`; if they agree, you've earned confidence the Normal was fine, and if they disagree, the robust one is almost certainly closer to the truth.

---

## 7. Ordinal and categorical outcomes

Two outcome types break the molds above and deserve their own treatment: **ordered** categories (a Likert rating 1–5, a severity grade, a class of degree) and **unordered** categories (which of three species, which of four brands). They are different problems and need different links.

### Ordinal regression: respect the order, don't fake it

A Likert response (1=strongly disagree … 5=strongly agree) is a number, but it is **not a quantity**. The gap between "disagree" and "neutral" is not guaranteed equal to the gap between "neutral" and "agree," and the categories have no meaningful zero or unit. Treating it as a continuous outcome (linear regression on the integer codes) silently assumes those gaps are equal and that predictions like $3.7$ are meaningful — both false. Treating it as an *unordered* categorical throws away the order information, which is real and valuable. The correct model is **ordinal regression**, and the standard form is the **cumulative logit**.

The idea is elegant. Imagine a latent continuous "agreeableness" $\eta_i = \mathbf{x}_i^\top\boldsymbol\beta$ for each respondent. We slice the latent line into $K$ ordered bins using $K-1$ **cutpoints** $\alpha_1 < \alpha_2 < \cdots < \alpha_{K-1}$. The observed category is whichever bin the latent value falls into. This gives cumulative probabilities:

$$
P(y_i \le k) = \sigma(\alpha_k - \eta_i),
$$

so increasing $\eta$ shifts the *entire* latent distribution toward higher categories at once. There is **no separate intercept** — the cutpoints absorb it (an intercept would just slide all cutpoints together and be unidentified).

```python
# Synthetic ordinal data with KNOWN cutpoints, so we can verify recovery.
K = 4
true_cut = np.array([-1.0, 0.5, 1.8])
N = 400
x_ord = rng.normal(0, 1, N)
true_b = 1.3
eta = true_b * x_ord
# Sample categories from the cumulative-logit model:
cum = 1 / (1 + np.exp(-(true_cut[None, :] - eta[:, None])))     # P(y <= k), shape (N, K-1)
probs = np.diff(np.c_[np.zeros(N), cum, np.ones(N)], axis=1)    # category probabilities
y_ord = np.array([rng.choice(K, p=p) for p in probs])          # ints in 0..K-1

with pm.Model() as ord_m:
    # Ordered cutpoints: the transform + sorted initval are REQUIRED or sampling starts invalid.
    cut = pm.Normal(
        "cutpoints",
        mu=np.linspace(-2, 2, K - 1), sigma=1.5,
        transform=pm.distributions.transforms.univariate_ordered,   # verify spelling in pinned PyMC
        initval=np.linspace(-2, 2, K - 1),
    )
    b = pm.Normal("b", 0, 1)            # NOTE: no intercept — cutpoints play that role
    eta_m = b * x_ord
    pm.OrderedLogistic("y", eta=eta_m, cutpoints=cut, observed=y_ord)  # outcome ints 0..K-1
    idata_ord = pm.sample(1000, tune=1000, chains=4, target_accept=0.95,
                          random_seed=RANDOM_SEED)
az.summary(idata_ord, var_names=["b", "cutpoints"], round_to=2)
```

Check recovery: `b` should land near the true $1.3$ and the cutpoints near $[-1.0, 0.5, 1.8]$. Because the outcome is on a latent logit scale, **interpret with care**: a positive `b` means higher $x$ pushes mass toward higher categories, but *how much* probability moves into each category depends on where the cutpoints sit — so push to the response scale (`compute_p=True` gives per-category probabilities) rather than reading `b` as a category change.

> ⚠️ **Pitfall — three ordinal footguns.** (1) **Code the outcome as integers `0..K-1`, not `1..K`** — `pm.OrderedLogistic` expects zero-based categories. (2) You **must** pass both `transform=...univariate_ordered` *and* a sorted `initval`; without them the cutpoints can start out of order, the initial log-probability is $-\infty$, and you get `SamplingError: Initial evaluation ... -inf` or a wall of divergences. (3) Confirm the transform spelling in your pinned PyMC — recent versions use `univariate_ordered`; older ones used `ordered`. Both have existed; prefer `univariate_ordered`.

In Bambi, ordinal models arrived in version 0.12.0 with two families:

```python
df_ord = pd.DataFrame({"y": y_ord, "x": x_ord})
model_ord = bmb.Model("y ~ x", data=df_ord, family="cumulative")  # cumulative logit (the standard)
# family="sratio" is the stopping/continuation-ratio alternative — same ordered outcome, different link
idata_ord_bmb = model_ord.fit(draws=1000, tune=1000, chains=4, target_accept=0.95,
                              random_seed=RANDOM_SEED)
```

> 🔧 **In practice:** Use `"cumulative"` for the standard cumulative-logit model and `"sratio"` (stopping ratio) when the process is sequential — "did you stop at category $k$ given you reached it?" — e.g. education levels you pass through in order, or staged failure. There is **no** `"ordinal"` or `"stoppingratio"` string; the keys are exactly `"cumulative"` and `"sratio"`. McElreath's `Trolley` data (moral-judgment ratings 1–7) is the classic real-world ordinal example if you want a non-synthetic one.

### Categorical / softmax regression: unordered classes

When the $K$ outcomes have **no order** — species, party, product chosen — we use the **categorical** family with a **softmax** link. We fix one class as the reference ($\beta_{\text{ref}} = 0$) and estimate a coefficient vector for each of the other $K-1$ classes. The probability of class $k$ is:

$$
P(y_i = k) = \frac{e^{\eta_{ik}}}{\sum_{j=1}^{K} e^{\eta_{ij}}}, \qquad \eta_{ik} = \mathbf{x}_i^\top\boldsymbol\beta_k, \quad \boldsymbol\beta_{\text{ref}} = \mathbf{0}.
$$

The softmax is the multi-class generalization of the sigmoid: it turns $K$ real-valued scores into $K$ probabilities that sum to 1.

```python
# Penguins: predict species (3 classes) from bill length & flipper length.
pen = sns.load_dataset("penguins").dropna().reset_index(drop=True)
pen["bill_z"]    = (pen.bill_length_mm    - pen.bill_length_mm.mean())    / pen.bill_length_mm.std()
pen["flipper_z"] = (pen.flipper_length_mm - pen.flipper_length_mm.mean()) / pen.flipper_length_mm.std()

model_cat = bmb.Model("species ~ bill_z + flipper_z", data=pen, family="categorical")
idata_cat = model_cat.fit(draws=1000, tune=1000, chains=4, target_accept=0.9,
                          random_seed=RANDOM_SEED)
az.summary(idata_cat, round_to=2)
```

The summary will show two sets of coefficients (for the two non-reference species), each on its own softmax-logit scale relative to the reference. As with every link-scale model, **the raw coefficients are not class-probability changes** — push to the response scale with `model_cat.predict(...)` or `bmb.interpret.predictions` to read "probability of Gentoo as bill length increases." The raw-PyMC equivalent uses `pm.Categorical` with `p = pm.math.softmax(eta, axis=-1)` where `eta` stacks a column of zeros (the reference) with the $K-1$ estimated linear predictors:

```python
# pseudocode shape of the raw-PyMC softmax model
# eta_k = X @ beta_k for k in non-reference classes; reference column pinned to 0
# p = pm.math.softmax(pt.concatenate([zeros_col, eta_nonref], axis=1), axis=1)
# pm.Categorical("species", p=p, observed=y_int)   # y_int in 0..K-1
```

> 💡 **Intuition — ordinal vs categorical is a modeling decision, not a data-type decision.** Both have integer-coded outcomes. The question is: *does the order carry information?* If moving from category 2 to 3 is "the same kind of step" as 3 to 4 (severity, agreement, rank), use **ordinal** — it shares one $\boldsymbol\beta$ across a latent line and spends $K-1$ cutpoints, far more efficient and honest. If the categories are just *different* (species, color, brand), use **categorical** — it spends a full coefficient vector per class. Choosing categorical for ordered data wastes information and overfits; choosing linear-on-the-codes invents a metric that isn't there. Match the model to the meaning.

---

## 8. Interactions: when the slope depends on something else

So far each predictor has had a single, fixed effect. **Interactions** let an effect *depend on another variable* — the slope of $a$ changes with the level of $b$. The canonical example, from McElreath, is so clean it has become a rite of passage: the effect of terrain **ruggedness** on a nation's GDP *flips sign* depending on whether the nation is in Africa. Outside Africa, rugged terrain hurts the economy (harder to trade, farm, build). Inside Africa, ruggedness is associated with *higher* GDP — historically, rugged regions resisted the slave trade and colonial extraction. One model, two opposite slopes. You cannot capture that with additive terms; you need an interaction.

```python
rugged = pd.read_csv(
    "https://raw.githubusercontent.com/rmcelreath/rethinking/master/data/rugged.csv", sep=";")
rugged = rugged.dropna(subset=["rgdppc_2000"]).copy()
rugged["log_gdp"]   = np.log(rugged["rgdppc_2000"])
rugged["log_gdp_s"] = rugged.log_gdp / rugged.log_gdp.mean()        # scale to mean 1 (McElreath)
rugged["rugged_s"]  = rugged.rugged / rugged.rugged.max()           # scale to [0,1]
rugged["rugged_c"]  = rugged.rugged_s - rugged.rugged_s.mean()      # center
rugged["cont_africa"] = rugged.cont_africa.astype(int)
rugged = rugged.reset_index(drop=True)
```

> 🔧 **In practice — a deliberate departure from z-scoring.** Everywhere else in this chapter we standardize (z-score) predictors by default (Chapter 03). Here we follow McElreath's exact rugged scaling instead — `log_gdp` divided by its mean (so the outcome is ~1 on average) and `rugged` divided by its max then centered — precisely so the priors below match his analysis. That is the *only* reason the intercept prior is `Normal(1, 0.1)`: the outcome is centered near 1, not 0. Z-scoring both variables would work just as well; it would only move the prior centers (intercept near 0, slopes in SD units). When you adopt someone else's scaling, adopt their priors with it — the two travel together.

The interaction term is the *product* of the two predictors. The linear predictor is:

$$
\mu_i = \beta_0 + \beta_R\,\text{rugged}_i + \beta_A\,\text{africa}_i + \beta_{RA}\,(\text{rugged}_i \times \text{africa}_i).
$$

The coefficient $\beta_{RA}$ on the product is **how much the ruggedness slope changes when `africa` goes from 0 to 1**. The slope of ruggedness is $\beta_R$ outside Africa and $\beta_R + \beta_{RA}$ inside it. If those have opposite signs, the slope flips.

### Raw PyMC

```python
coords = {"obs": rugged.index}
with pm.Model(coords=coords) as rugged_m:
    R = pm.Data("rugged_c", rugged.rugged_c.values, dims="obs")
    A = pm.Data("africa",   rugged.cont_africa.values, dims="obs")

    b0  = pm.Normal("Intercept", 1, 0.1)        # log_gdp_s is ~1 on average (we scaled it)
    bR  = pm.Normal("rugged",    0, 0.3)
    bA  = pm.Normal("africa",    0, 0.3)
    bRA = pm.Normal("rugged:africa", 0, 0.3)    # THE interaction coefficient
    sigma = pm.HalfNormal("sigma", 0.3)

    mu = pm.Deterministic("mu", b0 + bR * R + bA * A + bRA * (R * A), dims="obs")
    pm.Normal("log_gdp_s", mu=mu, sigma=sigma, observed=rugged.log_gdp_s.values, dims="obs")

    idata_rugged = pm.sample(1000, tune=1000, chains=4, target_accept=0.9,
                             random_seed=RANDOM_SEED)

# Recover the two slopes from the posterior:
post = az.extract(idata_rugged)
slope_outside = post["rugged"].values                       # beta_R
slope_inside  = post["rugged"].values + post["rugged:africa"].values   # beta_R + beta_RA
print("ruggedness slope OUTSIDE Africa:", round(slope_outside.mean(), 3))
print("ruggedness slope INSIDE  Africa:", round(slope_inside.mean(),  3))
```

You should see the outside-Africa slope come out **negative** and the inside-Africa slope **positive** — the famous sign flip, recovered with full posterior uncertainty on each. That, not the raw $\beta_{RA}$, is the number you report: "the association reverses direction."

### The same model in Bambi

```python
# a*b expands to a + b + a:b. Use a:b for the interaction term alone.
model_rugged = bmb.Model("log_gdp_s ~ rugged_c * cont_africa", data=rugged, family="gaussian")
idata_rugged_bmb = model_rugged.fit(draws=1000, tune=1000, chains=4, target_accept=0.9,
                                    random_seed=RANDOM_SEED)
```

The formula `rugged_c * cont_africa` **expands to** `rugged_c + cont_africa + rugged_c:cont_africa` — both main effects plus the interaction. If you ever want the interaction *without* a main effect (rare, and usually a mistake), `rugged_c:cont_africa` gives just the product term. This single-star expansion is the most useful piece of the formula grammar; it is why Bambi formulas read like the model you mean.

> ⚠️ **Pitfall — link-scale models are *already* non-additive on the outcome scale.** Here is a subtlety that trips up even experienced analysts. In a **logistic** or **Poisson** model, even *without* any interaction term, the effect of a predictor on the *response scale* (probability, count) **depends on the values of the other predictors** — because the inverse link (sigmoid, exp) is nonlinear. Two predictors that are additive on the log-odds scale are *not* additive on the probability scale. So you cannot read "no interaction term ⇒ effects don't interact" off a GLM. The honest way to ask "how does the effect of $x$ depend on $z$?" is to compute it on the response scale with marginal effects — which is exactly what we build next.

> 💡 **Intuition:** An interaction coefficient answers "how does one slope bend as another variable changes?" — but only on the *link* scale. For a human-facing answer ("how many more bikes per degree, on a humid day vs a dry day?") you must evaluate the model at concrete predictor values and difference the predictions on the outcome scale. Coefficients are for the model's internal bookkeeping; **predictions and contrasts are for people.**

---

## 9. Link scale vs outcome scale: always push to the human scale

This is the section that, if you take nothing else from the chapter, will keep you from shipping wrong numbers. **Coefficients live on the link scale; decisions live on the outcome scale.** Every GLM hands you parameters in log-odds or log-rate units that *no stakeholder thinks in*. Your job is to translate, and to carry the uncertainty across the bridge.

The Bayesian translation procedure is mechanical and exact — no delta method, no approximation:

1. Fix the predictors at the values you care about (a specific passenger, a counterfactual contrast, a grid).
2. Compute the linear predictor $\eta$ for **every posterior draw** of the coefficients.
3. Apply the inverse link $g^{-1}$ to **every draw** of $\eta$, giving a posterior of the response-scale quantity (probability, count).
4. Summarize *that* posterior (mean, 94% HDI). Difference two of them for a contrast.

Because you transform the whole posterior, the credible interval on the response-scale quantity is **exact** — it is just the spread of the transformed draws. Let me do it by hand for the Titanic logistic model, then show Bambi automate it.

### Counterfactual predictions by hand (raw PyMC)

We use the `pm.Data` containers we declared as *mutable* — swap in counterfactual predictor values, then call `sample_posterior_predictive`:

```python
# Counterfactual: probability of survival across ages, for a female 1st-class-fare passenger.
age_grid   = np.linspace(-2, 2, 50)                    # standardized ages
female_set = np.ones_like(age_grid)
fare_set   = np.full_like(age_grid, 1.0)               # ~1 SD above mean fare

with titanic_logit:
    pm.set_data({"female": female_set, "age_z": age_grid, "fare_z": fare_set})
    ppc = pm.sample_posterior_predictive(
        idata_logit, var_names=["p"], predictions=True,    # we saved Deterministic("p")
        random_seed=RANDOM_SEED, extend_inferencedata=False)

p_draws = ppc.predictions["p"]                # dims: (chain, draw, obs) = posterior of P(survive)
p_mean  = p_draws.mean(("chain", "draw")).values
p_hdi   = az.hdi(p_draws, hdi_prob=0.94)["p"].values   # (obs, 2): lower/upper band
# Plot p_mean with the HDI band against the unstandardized age axis -> a survival curve with uncertainty.
```

What you have built is a **survival-probability curve with a credible band** — for every age, the posterior probability that this passenger profile survives, with honest uncertainty. *That* is what a human reads. Notice you never touched a delta method; you pushed draws through the sigmoid and summarized. The same recipe gives counterfactual *counts* for a Poisson model (push through `exp`), counterfactual *category probabilities* for a softmax model, and so on.

> 🔧 **In practice:** Declaring your predictors as `pm.Data(...)` at model-build time is what makes this swap-in prediction possible — `pm.set_data` mutates them and the model recomputes. If you hard-code predictor arrays instead, you cannot do counterfactuals without rebuilding the model. Make `pm.Data` containers a habit for any predictor you might want to vary later. (This is also how out-of-sample prediction works in Chapter 11 and beyond.)

### The automated version: `bmb.interpret`

Doing the pushforward by hand is good for understanding and total control, but tedious for routine work. Bambi's `interpret` submodule automates the three canonical marginal-effects questions, computing each over the full posterior so the intervals are exact. It is the Bayesian analogue of R's **marginaleffects** package, and the design is consciously modeled on it.

**(a) Predictions** — "what is the expected response at these predictor values?"

```python
import bambi as bmb

# Adjusted predictions on the RESPONSE scale (probabilities), across age, holding others at defaults.
preds = bmb.interpret.predictions(model_logit, idata_logit_bmb,
                                  conditional={"age_z": np.linspace(-2, 2, 50),
                                               "female": [0, 1]})
preds.head()           # a DataFrame: predictor grid + posterior mean + lower/upper HDI on P(survive)

bmb.interpret.plot_predictions(model_logit, idata_logit_bmb,
                               conditional=["age_z", "female"])
```

`predictions` returns a tidy DataFrame of response-scale predictions with HDIs across the grid you specify in `conditional`; `plot_predictions` draws it. This is the survival curve we built by hand, in two lines, for *both* sexes at once.

**(b) Comparisons / contrasts** — "what is the *difference* (or *ratio*) in response when a predictor changes from one value to another?" This is where odds ratios and IRRs come from, automatically:

```python
# Probability DIFFERENCE of survival, female vs male — an average marginal effect on the prob scale.
comp_diff = bmb.interpret.comparisons(
    model_logit, idata_logit_bmb,
    contrast={"female": [0, 1]}, comparison="diff")     # "diff" -> probability difference
print(comp_diff)        # e.g. female raises P(survive) by ~0.5 (0.44, 0.56) — on the OUTCOME scale

# RATIO of odds -> the odds ratio directly, with an exact credible interval.
comp_ratio = bmb.interpret.comparisons(
    model_logit, idata_logit_bmb,
    contrast={"female": [0, 1]}, comparison="ratio")    # "ratio" -> odds ratio (logistic) / IRR (Poisson)
print(comp_ratio)
```

`comparison="diff"` gives the **average marginal effect on the response scale** — the honest "being female raises survival probability by about half" — which is what you tell a general audience. `comparison="ratio"` gives the **odds ratio** for a logistic model (or the **IRR** for a Poisson model) — the multiplicative statement for technical readers. Both come with exact credible intervals because they are computed over the posterior, not via a delta-method approximation. For the Poisson bikes model, `comparisons(..., contrast={"temp_z": [0, 1]}, comparison="ratio")` returns the incidence-rate ratio for a one-SD temperature increase, directly.

**(c) Slopes / marginal effects** — "what is $\frac{\partial(\text{response})}{\partial x}$ at given points?" — the instantaneous version, including elasticities:

```python
# dP(survive)/d(age) at each age — the local marginal effect of age on the probability scale.
slp = bmb.interpret.slopes(model_logit, idata_logit_bmb,
                           wrt="age_z", conditional={"age_z": np.linspace(-2, 2, 25)},
                           slope="dydx")               # "dydx" | "eyex"(elasticity) | "eydx" | "dyex"
bmb.interpret.plot_slopes(model_logit, idata_logit_bmb, wrt="age_z", conditional="age_z")
```

`wrt` is the variable you differentiate "with respect to"; `slope="dydx"` is the plain derivative, `"eyex"` an elasticity (percent-per-percent). Each `interpret` function returns a result object whose `.summary` is the DataFrame above and whose `.draws` holds the full posterior of the quantity — so you can compute *any* further summary you like.

> 💡 **Intuition — why marginal effects matter on a link-scale model.** Recall from §8 that even an interaction-free GLM is non-additive on the response scale. The slope of age on *probability* is not constant — it is steep near $p=0.5$ and flat near 0 or 1 (the sigmoid again). `slopes` evaluates that local steepness at each point, so it captures the *real*, position-dependent effect that the single coefficient `age_z` hides. This is the response-scale truth the divide-by-4 rule only approximates.

> 🔧 **In practice — verify the exact kwargs in your pinned Bambi.** The `interpret` API has drifted slightly across releases: `comparison=` (the singular spelling) is current, but older versions used `comparison_type=`; the `slope=` values `dydx / eyex / eydx / dyex` are confirmed. Check the installed version's signature before pasting — `help(bmb.interpret.comparisons)` is the fastest way. The three verbs — **predictions, comparisons, slopes** — and their meaning are stable even as kwarg spellings shift.

> 📜 **Citation/Origin:** Bambi's `interpret` submodule is modeled on Vincent Arel-Bundock's **marginaleffects** R/Python package, which standardized the "predictions / comparisons / slopes" vocabulary for marginal effects. The Bayesian twist — computing each quantity over posterior draws so intervals are exact rather than delta-method approximations — is a genuine advantage of doing this inside a probabilistic framework.

---

## 10. Worked end-to-end: a logistic model of Titanic survival, start to finish

Let me now chain every step of the workflow into one continuous example on the **Titanic** data, so you see the whole loop in one place rather than scattered across sections. We will build a logistic model, check the prior, fit, diagnose, posterior-predictive-check, and — the part that matters — extract decision-grade quantities on the outcome scale. We will run it in *both* Bambi and raw PyMC and confirm they agree.

### Step 0 — data and design

```python
titanic = sns.load_dataset("titanic")
df = titanic.dropna(subset=["age"]).copy()                  # Ch 16 will model the missing ages; here drop
df["female"] = (df["sex"] == "female").astype(int)
df["age_z"]  = (df.age  - df.age.mean())  / df.age.std()
df["fare_z"] = (df.fare - df.fare.mean()) / df.fare.std()
df["pclass"] = df["pclass"].astype("category")              # 1/2/3 -> treated as categorical
df = df.reset_index(drop=True)
```

Our question: holding fare and age fixed, how much did being female and travelling first class raise the *probability* of survival? Note "probability," not "log-odds" — we are designing the analysis around the outcome-scale quantity from the start.

### Step 1 — build the model and check the prior *before* fitting

```python
model = bmb.Model("survived ~ female + age_z + fare_z + C(pclass)",
                  data=df, family="bernoulli")
model.build()
model                                  # print: inspect the auto-chosen weakly-informative priors
# model.plot_priors()                  # draw them

# Prior predictive check: do the priors imply *plausible* survival rates before seeing y?
prior_pred = model.prior_predictive(draws=500, random_seed=RANDOM_SEED)   # InferenceData with `prior`
# az.plot_ppc(prior_pred, group="prior")  # simulated survival proportions should span (0,1) sanely
```

> 🩺 **Diagnostic — reading the prior predictive.** Before fitting, we ask: do my priors, *pushed through the model*, generate believable data? For a logistic model the prior-predictive survival *proportions* should spread reasonably across $(0,1)$ — not pile up at 0 and 1 (priors too wide, implying the model "expects" near-certain outcomes) and not collapse to exactly 0.5 (priors too tight). Bambi's auto-priors are tuned to pass this check; if you hand-set wild priors like `Normal(0, 100)` on standardized coefficients, the prior predictive will show the absurd bimodal-at-the-edges pattern that *is* the separation disease waiting to happen.

### Step 2 — fit and diagnose

```python
idata = model.fit(draws=1000, tune=1000, chains=4, target_accept=0.9,
                  random_seed=RANDOM_SEED,
                  idata_kwargs={"log_likelihood": True})

az.summary(idata, round_to=2)          # check r_hat <= 1.01, ess_bulk & ess_tail >= 400
# az.plot_trace(idata)                  # fuzzy caterpillars, no stuck chains
# az.plot_rank(idata)                   # uniform rank histograms = healthy mixing
```

Run the Chapter 07 gate: every $\hat R \le 1.01$, both ESS comfortably above 400, zero divergences. Logistic models on standardized predictors with weakly-informative priors essentially always sample cleanly — if this one *didn't*, your first suspicion would be separation (§3), checked with a crosstab. Here it will be green.

### Step 3 — posterior predictive check

```python
idata = model.predict(idata, kind="response", inplace=False)   # adds posterior_predictive
az.plot_ppc(idata, num_pp_samples=100)
# NOTE: kind="response" is required here. Bambi's predict DEFAULT is kind="response_params",
# which returns likelihood-PARAMETER draws (e.g. p), NOT the posterior_predictive group — so
# dropping the explicit kind= would silently leave posterior_predictive unpopulated and break plot_ppc.
# Test statistic: does the model reproduce the overall survival rate?
obs_rate = df.survived.mean()
rep_rate = idata.posterior_predictive["survived"].mean().item()   # mean over all reps + obs
print("observed survival rate: ", round(obs_rate, 3))
print("posterior-predictive survival rate:", round(rep_rate, 3))  # should sit on top of observed
```

For a binary outcome the most informative PPC is on **aggregate statistics**: the overall survival rate, and survival rates *within subgroups* (by sex, by class). A well-specified model reproduces all of them; a systematic miss in one subgroup (e.g. it gets overall survival right but underpredicts third-class survival) tells you a term is missing or a functional form is wrong. This is the model-criticism habit from Chapter 08 applied to a GLM.

### Step 4 — interpret on the outcome scale (the payoff)

Coefficients first, for the record, as **odds ratios**:

```python
post = az.extract(idata)
for name in ["female", "age_z", "fare_z"]:
    or_d = np.exp(post[name].values)
    lo, hi = az.hdi(or_d, hdi_prob=0.94)       # az.hdi for a genuine HDI on the skewed OR posterior
    print(f"OR[{name}] = {or_d.mean():.2f}  94% HDI [{lo:.2f}, {hi:.2f}]")
```

Then — the numbers a human actually wants — **probability differences** with exact intervals:

```python
# Average marginal effect of being female, on the PROBABILITY scale:
amf_female = bmb.interpret.comparisons(model, idata,
                                       contrast={"female": [0, 1]}, comparison="diff")
print(amf_female)        # "being female raises P(survive) by ~0.5 (HDI ...)"

# Counterfactual survival curve across age, for first-class females:
bmb.interpret.plot_predictions(model, idata, conditional=["age_z", "female"])
```

The deliverable is a sentence and a plot, both on the outcome scale: *"Holding age, fare, and class fixed, being female raised the probability of surviving by roughly 50 percentage points (94% HDI [...]); the effect of age was a modest decline in survival per year; first-class passengers survived at far higher rates than third-class."* No log-odds in sight — those stayed inside the model, where they belong.

### Step 5 — confirm Bambi and raw PyMC agree

For full transparency, here is the same model written out in raw PyMC with named coords for the class levels, so you can confirm the posteriors match Bambi's to within Monte Carlo error:

```python
pclass_idx, pclass_levels = pd.factorize(df["pclass"], sort=True)
coords = {"obs": df.index, "pclass": pclass_levels.astype(str)}
with pm.Model(coords=coords) as titanic_raw:
    female = pm.Data("female", df.female.values, dims="obs")
    age_z  = pm.Data("age_z",  df.age_z.values,  dims="obs")
    fare_z = pm.Data("fare_z", df.fare_z.values, dims="obs")
    cls    = pm.Data("cls", pclass_idx, dims="obs")

    b0   = pm.Normal("Intercept", 0, 1.5)
    bsex = pm.Normal("female",    0, 1.5)
    bage = pm.Normal("age_z",     0, 1.5)
    bfar = pm.Normal("fare_z",    0, 1.5)
    bcls = pm.ZeroSumNormal("pclass", sigma=1.5, dims="pclass")   # sum-to-zero class effects

    eta = b0 + bsex*female + bage*age_z + bfar*fare_z + bcls[cls]
    pm.Bernoulli("survived", logit_p=eta, observed=df.survived.values, dims="obs")
    idata_raw = pm.sample(1000, tune=1000, chains=4, target_accept=0.9,
                          random_seed=RANDOM_SEED)
```

The coefficients on `female`, `age_z`, `fare_z` will match Bambi's; the class encoding differs only by parameterization convention (Bambi uses treatment/dummy coding by default; here I used a sum-to-zero `ZeroSumNormal`, which is often better-identified). What is *invariant* to the coding is the part that matters: the **between-class contrasts** and the **predictions on the response scale** agree. The raw per-level class coefficients, however, are **not** directly comparable line-for-line, because the two codings put the reference in different places — treatment coding measures each class against the baseline class, while `ZeroSumNormal` measures each against the grand mean. So don't try to match coefficient tables cell-by-cell; compare predicted probabilities or contrasts (e.g. via `bmb.interpret.comparisons`) instead. The point stands: **Bambi wrote the PyMC model you would have written.** You are never running a black box.

---

## 11. The Bambi formula grammar, in one place

We've used Bambi piecemeal; here is the reference card. The formula syntax comes from the Wilkinson notation popularized by R's `lm`/`glm` and `lme4`, implemented for Bambi by the `formulae` library.

| You write | It means |
|---|---|
| `y ~ x` | intercept + slope on `x` |
| `y ~ 0 + x` | **no** intercept (suppress with `0 +` or `-1`) |
| `y ~ x + z` | additive: both main effects |
| `y ~ x:z` | interaction term **only** (the product) |
| `y ~ x*z` | expands to `x + z + x:z` (both mains + interaction) |
| `y ~ C(g)` | force `g` to be treated as categorical (dummy-coded) |
| `y ~ C(g, Treatment("ref"))` | categorical with a chosen reference level |
| `y ~ x + (1\|g)` | varying **intercept** by group `g` (random effect — Chapter 10) |
| `y ~ x + (x\|g)` | varying intercept **and slope** of `x` by group |
| `y['success'] ~ x` | Bernoulli: mark `'success'` as the event level (for string/multi-level outcomes) |
| `p(s, n) ~ x` | aggregated **Binomial**: `s` successes out of `n` trials |
| `y ~ x + offset(log(u))` | Poisson/NegBin **offset** (exposure `u`, coefficient pinned to 1) |
| `y ~ x + I(x**2)` | a transformed term (`I(...)` protects arithmetic) |

And the constructor and families, collected:

```python
bmb.Model(
    formula, data, family="gaussian", priors=None, link=None,
    categorical=None, dropna=False,
    auto_scale=True,           # weakly-informative priors scaled to the data (ON by default)
    center_predictors=True,    # center predictors internally for better geometry (ON by default)
    noncentered=True,          # non-centered parameterization for group effects (Ch 10)
)
```

Valid `family` strings (the keys you'll actually use): `"gaussian"`, `"t"`, `"bernoulli"`, `"binomial"`, `"poisson"`, `"negativebinomial"`, `"gamma"`, `"beta"`, `"wald"`, `"categorical"`, `"cumulative"` (ordinal), `"sratio"` (ordinal stopping-ratio). Valid `link` strings: `"identity"`, `"log"`, `"logit"`, `"probit"`, `"cloglog"`, `"inverse"`, `"inverse_squared"`, `"softmax"`. Each family has a sensible default link, so you rarely pass `link=` explicitly.

```python
idata = model.fit(
    draws=1000, tune=1000, chains=4, target_accept=0.9, random_seed=RANDOM_SEED,
    inference_method="pymc",                 # NUTS backends: "pymc", "nutpie", "numpyro", "blackjax"; also "vi", "laplace"
    idata_kwargs={"log_likelihood": True},   # needed for az.loo / az.compare (Ch 11)
)
```

> 🔧 **In practice:** `inference_method="numpyro"` routes sampling through the JAX/NumPyro backend for a large speedup on bigger models (Chapter 17), returning the same `InferenceData`. The full set of NUTS backends is `"pymc"`, `"nutpie"`, `"blackjax"`, and `"numpyro"`; `"vi"` and `"laplace"` give approximate inference. `inference_method="laplace"` gives a fast quadratic approximation (Chapter 04) for quick iteration before committing to full NUTS. (`help(model.fit)` on your pinned version lists the exact accepted strings.)

> 🔧 **In practice — there is no `"ordinal"` key.** Ordinal families are `"cumulative"` and `"sratio"` (added in Bambi 0.12.0). And the aggregated-binomial helper is `p(successes, trials)` — confirm against your version's docs if you see `proportion(...)` mentioned in older threads. When in doubt, the live API reference at `bambinos.github.io/bambi/api/Model.html` is canonical.

---

## ⚠️ Common errors & how to fix them

The GLM-specific trouble table. Keep it open while you fit. These build on the general sampling-diagnostic table from Chapter 07; here the causes are model-structural, not just geometric.

| Symptom (what you see) | Cause | Fix (in order) |
|---|---|---|
| Logistic coefficient runs to ±∞, huge `r_hat`, divergences, enormous posterior SD | **(Quasi-)separation**: a predictor perfectly splits the classes; a flat/wide prior gives no shrinkage to stop the climb | Standardize predictors; put a weakly-informative prior on coefficients — `StudentT(nu=4, 0, 1)` or `Cauchy(0, 2.5)` (Gelman 2008), or `Normal(0, 1.5)`. The prior **is** the regularizer. Check `pd.crosstab(x, y)` for a zero cell |
| Poisson/NegBin: `mu` overflows to `inf`, `nan` logp, sampler stuck at start | $\eta$ runs large and `exp(η)` overflows — usually too-wide log-scale priors and/or unstandardized predictors | Standardize predictors; keep log-scale priors **tight** (`Normal(0,1)` is already a 2.7× IRR/SD); anchor the intercept at `log(mean count)`; exponentiate exactly once inside the model |
| Poisson fits but PPC's simulated spread is too narrow / tail undershoots / `Var/Mean` ≫ 1 | **Overdispersion** — data variance exceeds the mean, violating Poisson equidispersion | Switch to `family="negativebinomial"` / `pm.NegativeBinomial`; confirm the gain with `az.compare` |
| Interpreted NegBin `alpha` as "more dispersion when large" and concluded "no overdispersion" | **α-vs-φ inversion** — PyMC's `alpha = 1/φ`; small `alpha` = MORE overdispersion | Re-read: in PyMC `Var = μ + μ²/α`, so small `alpha` = heavy overdispersion, `alpha → ∞` = Poisson. Triple-check the direction |
| Even NegBin undershoots the count of exact zeros | **Zero-inflation / hurdle** — a separate process generates structural zeros | `pm.ZeroInflatedPoisson` / `pm.ZeroInflatedNegativeBinomial` (Chapter 13) |
| Count coefficients implausibly large; a "size" variable dominates | **Forgot exposure** — modeling raw counts where rates were intended | Add `offset(log(exposure))` (Bambi) or `+ log_exposure` to η (PyMC); the offset's coefficient is pinned to 1 |
| Ordinal: `SamplingError: Initial evaluation ... -inf`, or a wall of divergences | Cutpoints not ordered at init; missing transform/initval; or outcome coded `1..K` | Use `transform=...univariate_ordered` **and** a sorted `initval`; code the outcome as ints `0..K-1` |
| Reporting log-odds/log-rate coefficients to stakeholders as if they were probability/count changes | Reading **link-scale** coefficients as **response-scale** effects | Report `exp(β)` (odds ratio / IRR) for technical readers; use `bmb.interpret` on the response scale and the divide-by-4 rule for everyone |
| `az.loo` / `az.compare` errors "no log_likelihood group" | Pointwise log-likelihood wasn't stored | Bambi: `fit(idata_kwargs={"log_likelihood": True})`; raw PyMC: `pm.compute_log_likelihood(idata, model=model)` |
| Bambi "could not determine success level" | Bernoulli outcome is a string / has >2 levels | Use the bracket syntax: `"y['success_level'] ~ ..."` |
| Bambi default priors "too tight," posterior barely moves vs your intuition | `auto_scale` scaled priors to **standardized** data; you compared to an unstandardized intuition; intercept is centered at the outcome mean | `model.plot_priors()` to look; override with `priors={...}` if you disagree; remember predictors are centered internally |
| StudentT robust model: `nu` wanders to huge values | Data have no real outliers; the t-likelihood is collapsing to Normal | This is *correct behavior* — it's telling you robustness wasn't needed. Nothing to fix |
| Probit vs logit coefficients differ by a constant factor and you panic | Probit and logit links have different scales (~1.6–1.8×) | Expected; compare on the **response scale**, not coefficient-to-coefficient |

---

## 🧪 Exercises

1. **(Conceptual — the recipe.)** For each of these outcomes, name the family, the canonical link, and one sentence on what a positive coefficient *means on the outcome scale*: (a) whether a loan defaults (yes/no); (b) the number of website visits per user per day; (c) a customer-satisfaction rating from 1 to 5; (d) which of four shipping carriers a customer picks. *Hint: two of these are easy to get wrong by treating an ordered or unordered category as a number.*

2. **(Coding — separation and the prior cure.)** Reproduce the separation demo in §3: build a perfectly-separating binary predictor, fit with a near-flat `Normal(0, 100)` prior and again with a `StudentT(4, 0, 2.5)` prior. Report, for the slope, the posterior mean, SD, $\hat R$, ESS, and divergence count under each prior. Write two sentences explaining *why* the prior — and nothing else — fixed it. *Hint: the likelihood under separation is a ramp with no peak; what does multiplying by a prior bump do to it?*

3. **(Coding — overdispersion, detected and fixed.)** On the `bikes` data, fit a Poisson and a negative-binomial model for `count ~ temp_z + humidity`. For each, run `az.plot_ppc` and compute the posterior-predictive distribution of the variance-to-mean ratio. Show the Poisson's predictive ratio fails to cover the observed value and the NB's covers it. Then run `az.compare` and report the ELPD difference and its standard error. State the posterior mean of NB `alpha` and translate it into "how much extra variance" using $\mathrm{Var}=\mu+\mu^2/\alpha$. *Hint: a small `alpha` means a lot of overdispersion — don't invert it.*

4. **(Coding — robustness.)** Take any clean linear dataset (or the §6 synthetic one). Fit Normal and Student-t models. Then inject outliers of growing size into a *single* point and, for each outlier magnitude, record the Normal slope, the Student-t slope, and the Student-t `nu` posterior mean. Plot all three against outlier magnitude. Explain the shape of each curve. *Hint: the Normal slope should drift; the t slope should stay put; `nu` should drop as the outlier grows.*

5. **(Coding — marginal effects on the outcome scale.)** On the Titanic logistic model from §10, use `bmb.interpret.comparisons` to compute (a) the **odds ratio** for being female (`comparison="ratio"`) and (b) the **probability difference** for being female (`comparison="diff"`). Then compute the probability difference *by hand* with `pm.set_data` + `sample_posterior_predictive` on the raw-PyMC model and confirm the two agree. Which number would you put in a slide for a non-technical audience, and why? *Hint: nobody outside a stats team thinks in odds ratios.*

6. **(Coding — the interaction sign flip.)** Fit the rugged × continent interaction model (§8) in Bambi with `log_gdp_s ~ rugged_c * cont_africa`. Use `bmb.interpret.plot_slopes` with `wrt="rugged_c"` conditioned on `cont_africa` to show the slope of ruggedness flips sign across the two groups, *with* credible bands. Confirm the two slopes match what you recover by adding the raw posterior coefficients ($\beta_R$ and $\beta_R+\beta_{RA}$). *Hint: the interaction coefficient is exactly the difference between the two slopes.*

7. **(Conceptual/coding — ordinal done right.)** Take any Likert-style ordinal outcome (synthesize one with known cutpoints if you have none). Fit it three ways: (a) linear regression on the integer codes, (b) unordered `family="categorical"`, (c) ordinal `family="cumulative"`. Compare predictions and `az.compare` ELPD. Argue from the results why (c) is the principled choice and what each of (a) and (b) gets wrong. *Hint: (a) invents equal spacing; (b) throws the order away and overfits with extra parameters.*

---

## 📚 Resources & further reading

The sources behind every model and interpretation in this chapter. If you read only two, make them McElreath Ch 10–12 and the Bambi `interpret` notebooks.

- **McElreath, *Statistical Rethinking* (2nd ed., 2020)** — Ch 10–11 (the GLM "maximum entropy" view, binomial/Poisson, log-odds and IRR), Ch 8 (the rugged interaction, our §8), Ch 12 (ordered-categorical "monsters and mixtures"). The intuition backbone. PyMC port: https://github.com/pymc-devs/pymc-resources (Rethinking_2).
- **Gelman, Hill & Vehtari, *Regression and Other Stories* (2020, free PDF)** — Ch 13–14 (logistic regression, the **divide-by-4 rule**), Ch 15 (Poisson, overdispersion, the arsenic-wells example that mirrors Bambi's slopes demo). The cleanest treatment of GLM interpretation in print. https://avehtari.github.io/ROS-Examples/
- **Gelman, Jakulin, Pittau, Su (2008), "A weakly informative default prior distribution for logistic and other regression models,"** *Annals of Applied Statistics* — the `StudentT(4, 0, 2.5)` default and the separation argument behind §3. https://arxiv.org/pdf/0901.4011
- **Nelder & Wedderburn (1972), "Generalized Linear Models,"** *JRSS-A* 135(3):370–384 — the original unifying paper; worth reading once for the "this was all one idea" jolt.
- **Bambi `Model` API reference** — exact constructor args, family/link strings, `.fit/.predict/.set_priors`. The authority for §11. https://bambinos.github.io/bambi/api/Model.html
- **Bambi — Logistic Regression (ANES) notebook** — `vote['clinton'] ~ party_id + party_id:age`, log-odds interpretation, `model.predict`; mirror it for any logistic walkthrough. https://bambinos.github.io/bambi/notebooks/logistic_regression.html
- **Bambi — `interpret`: Plot Predictions** https://bambinos.github.io/bambi/notebooks/plot_predictions.html · **Plot Comparisons** (odds ratios / IRR via `comparison="ratio"`) https://bambinos.github.io/bambi/notebooks/plot_comparisons.html · **Plot Slopes** (marginal effects, the arsenic example) https://bambinos.github.io/bambi/notebooks/plot_slopes.html — the marginal-effects engine of §9.
- **Bambi — Categorical Regression notebook** — `family="categorical"`, softmax, $K-1$ coefficient sets. https://bambinos.github.io/bambi/notebooks/categorical_regression.html
- **Bambi — Hierarchical Binomial notebook** — the `p(y, n)` aggregated-binomial syntax + `(1|g)`; the bridge to Chapter 10. https://bambinos.github.io/bambi/notebooks/hierarchical_binomial_bambi.html
- **Bambi 0.12.0 release notes** — introduces the `"cumulative"` and `"sratio"` ordinal families. https://github.com/bambinos/bambi/releases/tag/0.12.0
- **Capretto et al. (2022), "Bambi: A Simple Interface for Fitting Bayesian Linear Models in Python,"** *JSS* v103i15 — design rationale, the formula grammar, and the auto-prior construction. Cite for "why Bambi." https://www.jstatsoft.org/article/view/v103i15 (preprint arXiv:2012.10754)
- **PyMC example gallery — GLMs:** Binomial regression (`pm.math.invlogit`, aggregated `pm.Binomial`) https://www.pymc.io/projects/examples/en/latest/generalized_linear_models/GLM-binomial-regression.html · Negative-binomial regression (overdispersion + offset) https://www.pymc.io/projects/examples/en/latest/generalized_linear_models/GLM-negative-binomial-regression.html · Ordinal regression (`univariate_ordered` transform, Dirichlet-cutpoint alternative) https://www.pymc.io/projects/examples/en/latest/generalized_linear_models/GLM-ordinal-regression.html
- **PyMC distribution docs** — `pmc.NegativeBinomial` (the `(mu, alpha)` parametrization and `Var = μ + μ²/α` — anchor the α-vs-φ warning here) https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.NegativeBinomial.html · `pm.OrderedLogistic` (eta, cutpoints, integer outcome) https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.OrderedLogistic.html
- **PyMC Discourse — offset/exposure in Bambi** — the maintainer's `offset(log(...))` answer and its "lightly documented" caveat (our §4). https://discourse.pymc.io/t/how-to-specify-exposure-offset-in-a-bambi-model/13667
- **Martin, Kumar, Lao, *Bayesian Modeling and Computation in Python*, Ch 3 ("Linear Models and Probabilistic Programming")** — the course-aligned PyMC/Bambi GLM walkthrough. Free online: https://bayesiancomputationbook.com/notebooks/chp_03.html

---

## ➡️ What's next

You can now reach for the right family, link, and predictor for almost any outcome, write it in raw PyMC *or* Bambi, cure separation and overdispersion, resist outliers, and — most importantly — report effects on the scale a human can act on. But every model in this chapter assumed the observations were *exchangeable* — one flat pool, one set of coefficients. Real data are almost never that simple: passengers cluster in cabins, bikes in stations, students in schools, counties in states. **Chapter 10 — Hierarchical & Multilevel Models** takes the GLM skeleton you just mastered and lets its coefficients *vary by group*, sharing information across groups through partial pooling. The `(1|g)` and `(x|g)` formula terms we previewed here become the main event; the funnel and the non-centered reparameterization from Chapter 07 become daily tools; and the aggregated-binomial and Poisson patterns from this chapter become the per-group likelihoods. The GLM was the noun; hierarchy is the verb that brings it to life on structured data.
