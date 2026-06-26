# Chapter 08 — The Bayesian Workflow

### Turning seven chapters of tools into one disciplined method for not fooling yourself

> *A note from me to you.*
>
> *Up to now this course has handed you tools, one at a time, like a master mechanic laying instruments on a bench: probability as logic (Ch 01), priors and how to choose and check them (Ch 02–03), the analytic and grid and quadratic approximations (Ch 04), MCMC and NUTS (Ch 05), variational inference (Ch 06), and the full diagnostic suite (Ch 07). Each was sharp. But a bench of sharp tools is not a craft. The craft is knowing **which tool, in which order, and what to do when the thing you build is wrong** — because it will be wrong, the first several times, and the entire difference between an amateur and a professional is what happens next.*
>
> *This is the chapter where the tools become a method. We are going to study the **Bayesian workflow**: the iterative loop that takes you from a vague question and a pile of data to a model you can actually defend. I will give you the two canonical formulations of it — Gelman et al.'s sprawling, honest, descriptive "Bayesian Workflow" paper, and Betancourt's disciplined, prescriptive "Towards A Principled Bayesian Workflow" — and I will reconcile them into a single spine you can carry in your head. Then we will go deep, with runnable code, on the four checking tools that hold the whole thing together: **fake-data simulation and parameter recovery** (the single best debugging habit in all of Bayesian computation), **prior predictive checks**, **posterior predictive checks**, and **simulation-based calibration (SBC)**.*
>
> *Here is the mindset I want you to leave with. A Bayesian analysis is not a button you press; it is a conversation with a model you keep rebuilding. The model is a small world (McElreath's phrase from Ch 01), and the workflow is the disciplined set of habits that keep checking your small world against the large world it is supposed to represent. None of these checks proves your model is "true." What they do is something more useful: they make it **very hard to keep believing a model that is badly wrong.** That is the whole game. Let's learn to play it.*

---

## What you'll be able to do after this chapter

- **State the Bayesian workflow as an iterative loop**, not a checklist, and explain why iteration — fitting a *sequence* of models — is the point, not a failure.
- **Recite and reconcile** the two canonical workflows: Gelman et al. (2020) §2–§8 and Betancourt's four questions / fifteen steps / three phases — and collapse both onto a single shared spine.
- **Simulate fake data from known parameters, fit, and check recovery** — the cheapest, most reliable sanity check that your model and code are sane, with full PyMC v5 code.
- **Run a prior predictive check** (recapped from Ch 03) and read whether your priors imply physically plausible datasets *before* you ever touch real data.
- **Run a posterior predictive check** with `az.plot_ppc`, compute and interpret **Bayesian p-values** with `az.plot_bpv`, and use **LOO-PIT** to detect miscalibration that ordinary PPCs miss.
- **Understand simulation-based calibration** — why correct inference makes ranks **Uniform**, what ∪/∩/sloped rank histograms diagnose, and how to run SBC with `simuk` or hand-roll the loop.
- **Apply the folk theorem of statistical computing** — that bad sampling usually signals a bad model — and build *up* from a simple model, expanding only when a check demands it.
- **Walk a complete mini-workflow end to end** on a real dataset, visibly hitting every stage, so the loop becomes muscle memory.

A note on scope. This chapter is the **connective tissue** of the course. The *mechanics* of each tool live elsewhere and I will point you there constantly: divergences, $\hat R$, ESS, and the centered/non-centered cure are **Chapter 07 (MCMC Diagnostics & Debugging)**; prior design is **Chapter 03 (Priors 2: Choosing, Building & Checking)**; LOO/WAIC and `az.compare` are **Chapter 11 (Model Comparison & Evaluation)**; hierarchical funnels in the wild are **Chapter 10 (Hierarchical & Multilevel Models)**. My job here is to show you where every tool *sits in the loop* and how they chain. Think of this as the map, with the detailed surveys cross-referenced.

```python
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)
```

---

## 1. Why a *workflow* at all? The folk theorem and the iterative loop

Let me start with the single most important sentence anyone has written about applied Bayesian computation. It is Andrew Gelman's, and it is called **the folk theorem of statistical computing**:

> 📜 **Citation/Origin:** *"When you have computational problems, often there's a problem with your model."* — Gelman et al. (2020), "Bayesian Workflow," §5.1. The folk theorem says: when your sampler diverges, when $\hat R$ won't come down, when ESS collapses, the *first* hypothesis is not "the sampler is broken" — it is "the model is pathological." Computational trouble is usually the geometry of a bad model bleeding through.

Sit with that. In Chapter 07 you learned to read divergences as truth-tellers. The folk theorem is the same idea promoted to a *workflow principle*: the diagnostics are not a nuisance gate you have to clear; they are **information about your model**, and the right response to a failed diagnostic is usually to *change the model*, not just to crank `target_accept` to 0.99 and hope.

This is why we need a workflow rather than a recipe. A recipe is linear: do A, then B, then C, ship. A real Bayesian analysis is a **loop**:

```
        ┌──────────────────────────────────────────────────────────────┐
        │                                                              │
        ▼                                                              │
  domain knowledge ──▶ generative model ──▶ PRIOR PREDICTIVE CHECK     │
                                                   │                   │
                              (priors absurd? ─────┘ revise priors)    │
                                                   │                   │
                                                   ▼                   │
                                            fit (MCMC/NUTS)            │
                                                   │                   │
                                                   ▼                   │
                                     COMPUTATIONAL DIAGNOSTICS  ───────┤ folk theorem:
                                          (R̂, ESS, divergences)        │ bad fit ⇒
                                                   │                   │ fix the MODEL
                                                   ▼                   │
                                   POSTERIOR PREDICTIVE / RETRODICTIVE  │
                                                   │                   │
                              (misfits some feature? ─────────────────┤ expand model
                                                   │                   │
                                                   ▼                   │
                                 model expansion & comparison ─────────┘
                                                   │
                                                   ▼
                              (and validate the algorithm itself: SBC)
```

You enter at the top, and you go around — sometimes several times — adding a varying intercept here, a heavier-tailed likelihood there, a tighter prior, a reparameterization. **Each loop you make the model a little more like the data-generating process, and you only add complexity that a check demanded.** That last clause is the discipline. Anybody can build a baroque model; the skill is building the *simplest* model your checks will accept, and knowing exactly which check forced each piece of complexity.

> 💡 **Intuition:** The workflow is not "how to fit a model." It is **how to develop a model you can trust through a sequence of fits.** Gelman's phrase for this is the *topology of models* (§7.4) — you are not finding the one true model, you are traversing a space of related models, learning from each. Betancourt's phrase is *calibrating your inference before you believe it*. Same loop, two emphases.

> ⚠️ **Pitfall:** The most common mistake of strong ML practitioners coming to Bayes is to treat `pm.sample()` like `model.fit()` — run it once, read off the posterior mean, ship. `pm.sample()` *always returns something*, smooth and plausible-looking, even when the model is badly misspecified or the sampler never reached the typical set. The workflow is the set of habits that stops you from trusting that output blindly. Without it you have a very expensive way to compute confidently wrong numbers.

### Three reasons iteration is a feature, not a bug

1. **You cannot know the right model in advance.** Real data has structure you didn't anticipate — overdispersion, a fat tail, a subgroup that behaves differently. You discover this *by fitting and checking*, not by thinking harder up front. The model that survives is the one that has been beaten on.
2. **Simple models are debuggable; complex models are not.** If you start with the full hierarchical, zero-inflated, time-varying monster and it diverges, you have no idea which of twenty moving parts is the culprit. If you start simple and add one piece at a time, every failure is localized to the piece you just added. This is just good software engineering applied to models (Gelman §9, "modeling as software development").
3. **Checks at each stage catch *different* failures.** Prior predictive checks catch absurd priors. Computational diagnostics catch sampler/geometry failures. Posterior predictive checks catch likelihood misspecification. SBC catches subtle bugs in the inference itself that *every other check passes*. No single check is sufficient; the workflow layers them so that something is very likely to catch a given error.

---

## 2. The two canonical workflows, enumerated

There are two influential, careful, book-length treatments of the Bayesian workflow, and they are worth knowing by name because the rest of the field cites them constantly. They look different on the page. They are, underneath, the same loop seen from two temperaments. Let me lay both out honestly, then reconcile them.

### 2A. Gelman et al. (2020), "Bayesian Workflow" — the descriptive, many-model view

> 📜 **Citation/Origin:** Gelman, Vehtari, Simpson, Margossian, Carpenter, Yao, Kennedy, Gabry, Bürkner, Modrák, **"Bayesian Workflow"** (2020), arXiv:2011.01808 — a 77-page survey by essentially the entire Stan/ArviZ brain trust. It is not a tidy algorithm; it is an honest, sprawling field guide to *what experienced Bayesians actually do.* Its temperament is **descriptive and iterative**: here is the messy reality of fitting many models, and here is the toolbox.

The paper's structure *is* its argument. Here is the section map, lightly annotated — the bolded items are the ones this chapter teaches with code:

- **§2 Before fitting a model** — 2.1 choosing an initial model; 2.2 *modular* construction (build in pieces); 2.3 scaling and transforming parameters; **2.4 prior predictive checking**; 2.5 generative and partially-generative models.
- **§3 Fitting a model** — 3.1 initial values, adaptation, warmup; 3.2 how long to run; 3.3 approximate algorithms; **3.4 "Fit fast, fail fast"** (get a cheap, possibly-broken fit quickly so you learn fast).
- **§4 Using constructed/simulated data to find and understand problems** — **4.1 fake-data simulation**; **4.2 simulation-based calibration**; 4.3 experimentation with constructed data.
- **§5 Addressing computational problems** — **5.1 the folk theorem**; 5.2 "start simple *and* complex, meet in the middle"; 5.4 monitoring intermediate quantities; 5.5 stacking to reweight poorly-mixing chains; 5.6 multimodality and difficult geometry; **5.7 reparameterization**; 5.8 marginalization; 5.9 adding prior information; 5.10 adding data.
- **§6 Evaluating and using a fitted model** — **6.1 posterior predictive checking**; 6.2 cross-validation and influence of individual points; 6.3 influence of the prior; 6.4 summarizing inference and propagating uncertainty.
- **§7 Modifying a model** — 7.1 constructing a model for the data; 7.2 incorporating additional data; 7.3 working with priors; **7.4 a topology of models** (the space you traverse).
- **§8 Understanding and comparing multiple models** — 8.1 visualizing models relative to each other; 8.2 cross-validation and model averaging; 8.3 comparing many models.
- **§9 Modeling as software development** — version control, testing, reproducibility, readability.
- **§10 Worked example: golf putting** — the famous model-expansion story (logistic → first-principles geometry → new data → a physics model of "how hard the ball is hit" → a fudge factor). The single best illustration in the literature of *expanding a model because a check demanded it*.
- **§11 Worked example: planetary motion** — unexpected multimodality.
- **§12 Discussion** — iterative model building; "bigger datasets demand bigger models"; prediction, generalization, poststratification.

The thing to take from Gelman is the *spirit*: you fit **many** models, the diagnostics are information, and model-building is an iterative, partly-improvised craft with a well-stocked toolbox. The golf-putting example (§10) is worth reading on its own — it is the canonical narrative of starting with a thoughtless logistic, watching it fail a check, and being *led by the data* through a sequence of better models to a genuinely mechanistic one.

### 2B. Betancourt, "Towards A Principled Bayesian Workflow" — the prescriptive, calibrate-first view

> 📜 **Citation/Origin:** Michael Betancourt, **"Towards A Principled Bayesian Workflow"** (betanalpha.github.io case study). Its temperament is the opposite of Gelman's: **prescriptive and sequential**. It is a *disciplined checklist* organized around four evaluating questions and fifteen concrete steps, with a striking insistence: **calibrate your algorithm and your model on simulated data before you are ever allowed to look at the real data.**

Betancourt organizes everything around **four evaluating questions** — the questions a finished analysis must be able to answer:

1. **Question One — Domain Expertise Consistency.** Are the model's assumptions consistent with what you actually know about the domain? *Answered by prior pushforward / prior predictive checks.*
2. **Question Two — Computational Faithfulness.** Can your algorithm actually fit this model? *Answered by SBC and computational diagnostics.*
3. **Question Three — Inferential Adequacy.** Will the inferences be precise enough to be useful for the application? *Answered by checking posterior sensitivity / whether there is enough information.*
4. **Question Four — Model Adequacy.** Does the model capture the real data-generating structure? *Answered by posterior **retrodictive** checks against the observed data.*

> 📜 **Citation/Origin:** Betancourt deliberately says **"retrodictive"** rather than "predictive" for the final check. The model isn't predicting the future; it is *retrodicting* data you already hold. Use his term when you mean "posterior predictive check against the in-sample data" — it is more precise, and it signals you know the literature.

And these four questions unfold into **three phases and fifteen steps**:

- **Phase — Pre-Model, Pre-Data:** (1) Conceptual Analysis, (2) Define Observational Space, (3) Construct Summary Statistics.
- **Phase — Post-Model, Pre-Data:** (4) Model Development, (5) Construct Summary Functions, (6) Simulate Bayesian Ensemble, (7) Prior Checks, (8) Configure Algorithm, (9) Fit Simulated Ensemble, (10) Algorithmic Calibration (SBC over the ensemble), (11) Inferential Calibration.
- **Phase — Post-Model, Post-Data:** (12) Fit the Observation, (13) Diagnose Posterior Fit, (14) Posterior Retrodictive Checks, (15) Celebrate.

Notice where the real data enters: **step 12 of 15.** Eleven of Betancourt's fifteen steps happen *before you fit the observed data at all.* You build the model, simulate ensembles of fake data from it, check the priors, configure the sampler, fit the *simulated* data, calibrate the algorithm with SBC, and check that inferences would be adequate — all on data you generated yourself. Only then, with a model and an algorithm you have *proven* can recover truth, do you turn the crank on reality.

> 💡 **Intuition:** Gelman is the experienced surgeon improvising in a messy operating room with a deep toolbox. Betancourt is the airline pilot's pre-flight checklist: every item verified on the ground before you are allowed off it. Neither is wrong. Betancourt's discipline is what you want for a high-stakes model you will deploy and defend; Gelman's iterative pragmatism is what every real project actually feels like day to day. The mature practitioner runs Gelman's loop *with Betancourt's discipline at the gates.*

### 2C. The reconciliation — one spine

Strip away the temperamental differences and both authors describe the **same spine**:

```
generative model → PRIOR PREDICTIVE CHECK → fit → COMPUTATIONAL DIAGNOSTICS
                 → POSTERIOR PREDICTIVE (retrodictive) CHECK → EXPAND / COMPARE
   (with FAKE-DATA RECOVERY and SBC sitting between "can I fit it" and "fit the real data")
```

Here is the explicit crosswalk. Build this table into your mental model:

| Stage of the spine | Gelman et al. (2020) | Betancourt (principled) | This course |
|---|---|---|---|
| Frame the problem | §2.1 choosing initial model | Step 1 Conceptual Analysis; Q1 | Ch 01, this ch §3 |
| What can we observe | §2.5 generative models | Steps 2–3 observational space + summary stats | Ch 01, this ch §4 |
| Build the model | §2.2 modular; §2.3 scaling | Step 4 Model Development | Ch 02–03, 09–16 |
| Check priors *before* data | **§2.4 prior predictive checking** | Steps 6–7 simulate ensemble + Prior Checks; Q1 | **Ch 03**, this ch §6 |
| Can we even fit it? | §3; §3.4 fail fast | Step 8 Configure Algorithm; Q2 | Ch 05, this ch §5,§8 |
| Validate the *algorithm* | **§4.1 fake data; §4.2 SBC** | Steps 9–10 fit ensemble + Algorithmic Calibration; Q2 | **this ch §5, §8** |
| Will inference be useful? | §6.4 summarize/propagate | Step 11 Inferential Calibration; Q3 | Ch 11, this ch §9 |
| Fit the real data | §3 | Step 12 Fit the Observation | this ch §9 |
| Diagnose computation | **§5 (folk theorem, reparam)** | Step 13 Diagnose Posterior Fit | **Ch 07**, this ch §7 |
| Check fit to data | **§6.1 posterior predictive checking** | Step 14 Posterior Retrodictive Checks; Q4 | **this ch §7** |
| Improve / compare | §7 modifying; §8 comparing | (loop back; expand) | **Ch 11**, this ch §10 |
| Engineering hygiene | §9 modeling as software | (implicit) | this ch §11 |

> 🔧 **In practice:** You do **not** need to memorize fifteen steps. Memorize the **spine** — six boxes — and treat Gelman's toolbox and Betancourt's checklist as two zoom levels on it. When a project gets serious, zoom in to Betancourt's discipline (especially "calibrate before you fit real data"). When you're exploring, run Gelman's fast loop. The rest of this chapter teaches the four *checking* tools that turn the spine from a diagram into a practice.

---

## 3. Fake-data simulation & parameter recovery — the habit that will save you most

If you adopt exactly one habit from this entire chapter, make it this one. Before you fit a model to your real, precious, hard-won data, **fit it to fake data you generated yourself from known parameters, and check that the posterior recovers those parameters.** It is the cheapest, fastest, most reliable way to catch the bug that otherwise costs you a week.

Here is the logic, and it is airtight. Your model defines a joint distribution $p(y, \theta) = p(y\mid\theta)\,p(\theta)$ — which means it is also a *simulator*. You can run it forward: pick a specific true parameter vector $\theta^\*$, draw a dataset $y^\* \sim p(y \mid \theta^\*)$, and now you have a dataset whose true generating parameters you *know exactly*. Feed that dataset back into the model, condition on it, and look at the posterior $p(\theta \mid y^\*)$. If your model and your code are correct, that posterior must concentrate around $\theta^\*$. If it doesn't — if the true value is in the far tail, or outside the 94% interval, or the posterior is centered somewhere else entirely — you have found a bug *before it cost you anything.*

> 💡 **Intuition:** Real data never tells you whether you got the right answer, because you don't know the right answer — that's why you're fitting a model. Fake data is the one situation in all of statistics where **you know the truth**, so it is the one situation where you can directly check whether your inference machinery works. Use it relentlessly. McElreath simulates from his models constantly for exactly this reason; it is generative thinking (Ch 01) turned into a debugging tool.

### What parameter recovery catches

This single check catches a remarkable range of bugs, because to *pass* it, three completely separate things all have to be right:

1. **Your model is coded correctly** — the right distributions, the right link function, indices and shapes lined up, the design matrix wired to the right coefficients.
2. **Your inference runs correctly** — the sampler reaches the typical set, mixes, and the posterior actually concentrates where the likelihood and prior say it should.
3. **Your model is, in principle, *identifiable*** — the data you're generating actually contains enough information to pin down the parameters. If two parameters are confounded, recovery fails *even with perfect code*, and that is itself an important discovery about your design.

A sign flip in a coefficient, a `sigma` where you meant `sigma**2`, an off-by-one in a group index, a likelihood that's secretly the wrong family — all of these show up immediately as "the truth is not where the posterior is." Real data would have *hidden* every one of them behind a plausible-looking but wrong answer.

### A full worked recovery — linear regression

Let's do it properly. We'll use a simple linear regression so the focus stays on the *workflow*, not the model. First, **simulate from a known generative process** (the bible explicitly encourages synthetic-with-known-truth for exactly this teaching purpose):

```python
# --- Step 1: choose TRUE parameters and simulate a dataset from them ---
a_true, b_true, sigma_true = 2.0, 0.5, 1.0     # the truth we will try to recover
N = 100
x = rng.normal(size=N)                          # a predictor
mu_true = a_true + b_true * x
y = rng.normal(mu_true, sigma_true)             # y ~ Normal(a + b x, sigma)

print(f"true a={a_true}, b={b_true}, sigma={sigma_true}")
```

Now write **the exact model you intend to use on real data** — not a special "testing" model, the real one — and fit it to the fake `y`:

```python
# --- Step 2: fit the model you actually intend to use ---
coords = {"obs_id": np.arange(N)}
with pm.Model(coords=coords) as recovery_model:
    x_data = pm.Data("x", x, dims="obs_id")     # mutable in v5; swap-in for prediction later
    a = pm.Normal("a", mu=0, sigma=5)
    b = pm.Normal("b", mu=0, sigma=2)
    sigma = pm.HalfNormal("sigma", sigma=2)
    mu = pm.Deterministic("mu", a + b * x_data, dims="obs_id")
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y, dims="obs_id")

    idata = pm.sample(1000, tune=1000, chains=4, target_accept=0.9,
                      random_seed=RANDOM_SEED)
```

Now the moment of truth. Check that the posterior covers the truth:

```python
# --- Step 3: did we recover the truth? ---
az.summary(idata, var_names=["a", "b", "sigma"])
```

You will get something like this (your exact digits depend on the seed, but the structure will match):

| | mean | sd | hdi_3% | hdi_97% | ess_bulk | ess_tail | r_hat |
|---|---|---|---|---|---|---|---|
| a | 2.03 | 0.10 | 1.84 | 2.22 | 4100 | 3000 | 1.00 |
| b | 0.47 | 0.10 | 0.28 | 0.66 | 4300 | 2900 | 1.00 |
| sigma | 1.01 | 0.07 | 0.88 | 1.15 | 3800 | 2800 | 1.00 |

Read it like a verdict. The true `a=2.0` sits at the posterior mean `2.03`, comfortably inside `[1.84, 2.22]`. True `b=0.5` is inside `[0.28, 0.66]`. True `sigma=1.0` is inside `[0.88, 1.15]`. **Every true value is well within its 94% HDI.** And the diagnostics are clean — $\hat R = 1.00 \le 1.01$, ESS in the thousands. This model and this code can recover truth. *That* is what earns you the right to fit real data.

> 🩺 **Diagnostic:** The right way to read a recovery summary is, for each parameter, ask "is the true value inside the credible interval, and not way out in the tail?" A truth at the posterior mean is ideal; a truth near the edge of the 94% interval on *one* parameter once in a while is fine (a 94% interval misses the truth ~6% of the time by construction — that's what 94% means). A truth *outside* the interval, especially the same parameter every time you re-seed, is a bug. The rigorous version of "does this happen at the right rate?" is SBC (§8) — recovery is the one-shot, eyeball version; SBC is the many-shot, statistical version.

### Make the verdict visual and automatic

Eyeballing a table is fine for three parameters; for twenty it isn't. Overlay the truth on the marginal posteriors with `az.plot_posterior` and its `ref_val` argument — a vertical line at the true value that should land inside the fat part of each posterior:

```python
az.plot_posterior(
    idata, var_names=["a", "b", "sigma"],
    # dict form is name-keyed: it matches each truth to its variable by NAME,
    # so it is order-independent and survives non-scalar/extra variables.
    ref_val={"a": a_true, "b": b_true, "sigma": sigma_true},
)
```

Use the **dictionary** form of `ref_val` (name → value), not a bare list. The list form is matched to variables *positionally* and will silently mis-align — or break — the moment a variable is non-scalar or the flattening order differs; the dict form is keyed by name and therefore robust. Each panel shows a posterior with a vertical reference line at the truth. **What good looks like:** the line sits under the bulk of the density, near the middle. **What a bug looks like:** the line is out in a tail, or off the plotted range entirely. For a programmatic check you can compute coverage directly:

```python
# Automated recovery check: is each truth inside its 94% HDI?
post = idata.posterior
for name, truth in [("a", a_true), ("b", b_true), ("sigma", sigma_true)]:
    lo, hi = az.hdi(idata, var_names=[name], hdi_prob=0.94)[name].values
    inside = lo <= truth <= hi
    print(f"{name:5s} truth={truth:+.2f} 94% HDI=[{lo:+.2f}, {hi:+.2f}]  "
          f"{'OK' if inside else '*** MISSED ***'}")
```

In a test suite (Gelman §9, "modeling as software development"), you'd turn that into an `assert`. Parameter recovery is the unit test of Bayesian modeling.

> ⚠️ **Pitfall — recovering with the wrong sample size.** Recovery can *succeed* with `N=100000` and *fail* with `N=20`, for the same correct model, simply because tiny data doesn't pin the parameters down — the posterior is correctly wide and the truth wanders near the edges. So run recovery at a sample size *comparable to your real data*. If recovery is shaky at your real `N`, that's not necessarily a bug; it may be telling you the honest truth that **your data cannot answer your question precisely**, which is exactly the kind of thing Betancourt's "inferential adequacy" (Q3) is meant to surface *before* you over-interpret a noisy real fit.

> ⚠️ **Pitfall — recovering once proves little about a hard model.** A single recovery on an easy model is reassuring. A single recovery on a *funnel-prone hierarchical model* can pass by luck even when the sampler is biased, because one draw of the truth might land in an easy region. For anything with tricky geometry, escalate from one-shot recovery to full **SBC** (§8), which repeats the recovery over the whole prior and checks the *rate* of coverage statistically.

---

## 4. Prior predictive checks — does your model imply sane *data*?

You met prior predictive checks in **Chapter 03 (Priors 2)**, where we used them to *choose* priors. Here I want to recap them in their role as a **workflow stage** — the gate you pass *before* fitting, the answer to Betancourt's Question One (domain expertise consistency). The idea is short and powerful, so let me restate it cleanly and give you the v5 code one more time, because you will run this on every serious model you ever build.

A prior predictive check asks: **forget the data for a moment — what datasets does my model think are plausible, given only the priors?** Formally, you sample from the **prior predictive distribution**

$$ p(\tilde y) = \int p(\tilde y \mid \theta)\, p(\theta)\, d\theta, $$

in ASCII: `prior_predictive(ỹ) = ∫ likelihood(ỹ | θ) × prior(θ) dθ`. Operationally this is a *forward simulation*: draw $\theta \sim p(\theta)$ from the prior, then draw $\tilde y \sim p(\tilde y \mid \theta)$ from the likelihood, and repeat. Each iteration gives you a complete *fake dataset* that your model considers a priori plausible. If those fake datasets are physically absurd — human heights of $10^6$ cm, count data in the trillions, probabilities that are always essentially 0 or 1 — then your priors are encoding nonsense, and **no amount of data will fully undo a prior that rules out the truth.** Catch it now.

> 💡 **Intuition:** Priors are almost impossible to judge on the *parameters* directly — what *is* a reasonable prior on a logistic-regression coefficient? Nobody has intuition for log-odds. But everybody has intuition for **data**: you know what human heights look like, what plausible reaction times are, how many bike rentals a city sees in an hour. The prior predictive check *pushes your priors forward through the model into the space of data*, where your domain intuition actually lives, and lets you judge them there. This is Betancourt's "prior pushforward."

### The v5 code

In PyMC v5, prior predictive sampling uses `pm.sample_prior_predictive` and stores results in the `prior` and `prior_predictive` groups of the InferenceData:

```python
with recovery_model:
    idata_prior = pm.sample_prior_predictive(draws=500, random_seed=RANDOM_SEED)
# draws live in idata_prior.prior (parameters) and idata_prior.prior_predictive (fake y)
```

> ⚠️ **Pitfall — the argument is `draws`, not `samples`.** In current PyMC v5, `pm.sample_prior_predictive` takes `draws=`. Older code (and some tutorials) used `samples=`, which will error on a modern install. (If you hit a version that wants `samples=`, the dossier flags this as version-dependent — `# verify in current PyMC docs`.) Either way, the result lands in `idata.prior` / `idata.prior_predictive`, not in `idata.posterior`.

Now *look* at the implied data. ArviZ plots a prior predictive check with the very same `az.plot_ppc` you use for posterior checks — you just point it at the `prior` group:

```python
az.plot_ppc(idata_prior, group="prior", num_pp_samples=200)
```

> 🩺 **Diagnostic — reading a prior predictive plot.** Each thin line is one fake dataset's density, simulated purely from the priors; together they show the *range of datasets your model finds plausible before seeing anything.* **What good looks like:** the cloud of simulated datasets brackets the scale and shape of data you'd consider physically possible — wide enough to be honest about your uncertainty, but not absurd. **What bad looks like:** simulated `y` ranging over many orders of magnitude beyond anything real, or piling up at an impossible boundary. For our standardized linear model with `a ~ Normal(0, 5)`, `b ~ Normal(0, 2)`, `sigma ~ HalfNormal(2)` and a roughly standard-normal `x`, the implied `y` should span a believable handful of units — weakly informative, not insane. If instead you'd written `a ~ Normal(0, 1000)`, the prior predictive would explode, and the plot would tell you instantly.

### A quantitative prior predictive check

Plots are great; numbers are better for the things you can name. Pull the raw prior-predictive draws and check a domain-meaningful summary against what you know:

```python
pp = idata_prior.prior_predictive["y_obs"].values  # shape: (chain, draw, obs_id)
print(f"prior-predictive y range: [{pp.min():.1f}, {pp.max():.1f}]")
print(f"prior-predictive y mean ± sd: {pp.mean():.2f} ± {pp.std():.2f}")
```

If you were modeling, say, adult human heights and this printed a range of `[-400, 600]` cm, you'd know immediately to tighten the prior — negative heights are impossible and 6-metre humans don't exist. The fix is almost always the same trio you learned in Ch 03: **standardize your predictors** (so a unit-scale slope prior is sane), put **weakly-informative priors on a sensible scale**, and re-run the check until the implied data looks like data.

> 🔧 **In practice:** Prior predictive checking is not a one-time ritual; it's a tuning loop *inside* the workflow loop. You'll often iterate priors → prior predictive → priors a few times before you ever call `pm.sample()`. That iteration is cheap (no MCMC required, just forward simulation) and it pays for itself many times over by the divergences and pathologies it prevents downstream — recall the folk theorem: bad geometry often starts as a bad (too-vague) prior.

---

## 5. Fit, then gate on computational diagnostics (the Chapter 07 stage)

The next two stages of the spine — **fit** and **computational diagnostics** — are the entire subject of **Chapter 07 (MCMC Diagnostics & Debugging)**, so I will not re-derive them. But I want to put them in their place in the loop and remind you of the *gate*, because the workflow's logic hinges on it.

You fit (Gelman §3, Betancourt step 12). The recommended posture, especially early, is Gelman's **"fit fast, fail fast"** (§3.4): get a cheap, possibly-broken fit quickly so you *learn* quickly. A first pass with default settings tells you in two minutes whether the model is even in the right universe; you don't need 10,000 polished draws to discover that your indices are scrambled.

Then you **gate**. Before you read a single posterior number, you check that the fit is trustworthy. This is the loop you drilled in Chapter 07; here it is as the workflow's quality gate:

```python
# The computational-diagnostics gate (mechanics: Chapter 07)
n_div = int(idata.sample_stats["diverging"].sum())
print(f"divergences: {n_div}")            # target: 0
az.summary(idata, var_names=["a", "b", "sigma"])  # scan r_hat <= 1.01, ess >= 400
az.plot_trace(idata, var_names=["a", "b", "sigma"])  # fuzzy caterpillars, chains overlap
az.plot_energy(idata)                     # BFMI > 0.3; the two energy densities overlap
```

The thresholds, quoted consistently across this course (defined in Chapter 07):

| Diagnostic | Healthy | Trouble | Course threshold |
|---|---|---|---|
| Divergences | 0 | any | **0 is the target** |
| $\hat R$ (rank-normalized split) | $\le 1.01$ | $> 1.01$ | **$\le 1.01$** |
| ESS (bulk and tail) | $\gtrsim 400$ total | small (e.g. 30) | **$\gtrsim 400$ (≈100/chain)** |
| BFMI (energy) | $> 0.3$ | $< 0.3$ | **$> 0.3$** |

> 🩺 **Diagnostic:** If this gate is red, **the folk theorem tells you to suspect the model, not the sampler.** The canonical example — divergences in a hierarchical funnel cured by the centered → non-centered reparameterization — is Chapter 07's flagship, and you will reach for it again in **Chapter 10**. Raising `target_accept` to 0.95–0.99 is a legitimate first move for a *few* divergences, but if it doesn't clear them, stop tuning and start changing the model: reparameterize (Gelman §5.7), tighten a scale prior, marginalize a discrete latent (§5.8). A clean gate is the *entry ticket* to the next stage. You are not allowed to interpret a posterior whose fit failed the gate, because that posterior may be a systematically biased exploration of the wrong region of parameter space.

> ⚠️ **Pitfall:** Do not "diagnose around" a failure by deleting the divergent draws, or by running one long chain so $\hat R$ can't be computed (it needs $\ge 2$ chains), or by lowering your standards to $\hat R \le 1.1$. These are ways of *hiding* the problem, not fixing it. The gate exists to protect you from yourself.

---

## 6. Posterior predictive / retrodictive checks — does the *fitted* model reproduce the data?

A clean diagnostic gate tells you that you fit *the model you wrote* correctly. It says **nothing** about whether the model you wrote is any good. A perfectly-sampled posterior from a badly misspecified likelihood is a clean fit of a wrong model. The tool that asks "is the model *right enough*?" is the **posterior predictive check (PPC)** — Gelman §6.1, Betancourt's step 14 (his "posterior **retrodictive** check"), the answer to Question Four (model adequacy).

The idea mirrors the prior predictive check, but now we condition on the data. We draw from the **posterior predictive distribution**

$$ p(\tilde y \mid y) = \int p(\tilde y \mid \theta)\, p(\theta \mid y)\, d\theta, $$

ASCII: `posterior_predictive(ỹ | y) = ∫ likelihood(ỹ | θ) × posterior(θ | y) dθ`. Operationally: for each posterior draw $\theta^{(s)}$, simulate a whole fake dataset $\tilde y^{(s)} \sim p(\tilde y \mid \theta^{(s)})$. That gives you a population of datasets the *fitted* model considers plausible. Then you ask the question that defines model checking: **does the real dataset look like a typical draw from this population, or like an outlier?** If the real data has features your model's fake data never reproduces, the model is missing something real.

> 💡 **Intuition:** A good model is one that, after being told about the data, can *generate data like the data.* If it can't — if your model's simulated datasets are systematically too narrow, or never have as many zeros, or never reach the extremes the real data hits — then the model has failed to capture some feature of reality, and the PPC will show you *which* feature so you know what to add next. This is the check that drives Gelman's golf-putting expansion: a PPC fails, you see *how* it fails, and the failure tells you what physics to add.

### Running a PPC in PyMC v5 + ArviZ

First, generate posterior predictive draws (this writes the `posterior_predictive` group in place):

```python
with recovery_model:
    pm.sample_posterior_predictive(idata, extend_inferencedata=True,
                                   random_seed=RANDOM_SEED)
```

Then the workhorse overlay plot:

```python
az.plot_ppc(idata, num_pp_samples=100)     # kind="kde" by default
```

> 🩺 **Diagnostic — reading `az.plot_ppc`.** You get a thick **observed** density (your real `y`, a KDE) and many thin **posterior-predictive** densities (one per sampled dataset), plus usually a dashed **posterior-predictive mean**. **What good looks like:** the observed curve sits comfortably inside the cloud of predictive curves, threading the middle of the spaghetti — the model generates data like your data. **What bad looks like:** the observed curve walks *outside* the predictive cloud somewhere — a fatter tail than the model can produce, a bimodality the model smooths over, a spike at zero (excess zeros → you need a zero-inflated or hurdle model, Chapter 13), or skew the symmetric likelihood can't match. The *location* of the mismatch is the diagnosis. For our well-specified linear model fit to data we generated from that same model, the observed KDE should sit squarely inside the cloud — as it must, since the model is literally correct here.

`az.plot_ppc` has a few knobs worth knowing: `kind="cumulative"` overlays ECDFs (often more sensitive to tail misfit than KDEs because it doesn't smooth), `kind="scatter"` shows individual points, and `data_pairs={"y_obs": "y_obs"}` maps your observed variable to its predictive counterpart when the names differ.

### Test statistics and Bayesian p-values — checking *specific* features

The overlay plot is a *global* check of the whole distribution. Often you care about a **specific feature** — the maximum, the number of zeros, the standard deviation, the skew — and you want a quantitative verdict on whether the model reproduces it. That's a **Bayesian p-value**. Pick a test statistic $T(\cdot)$, compute it on the observed data $T(y)$ and on every predictive dataset $T(\tilde y^{(s)})$, and ask how often the predictive value exceeds the observed:

$$ p_B = \Pr\!\big(T(\tilde y) \ge T(y) \,\big|\, y\big). $$

> 🧮 **The math:** A *well-calibrated* feature gives $p_B \approx 0.5$ — the observed statistic sits right in the middle of the predictive distribution of that statistic, equally likely to be exceeded or not. A value near **0 or 1** is a red flag: the observed feature is in the extreme tail of what the model predicts, i.e. the model *systematically* mis-reproduces that feature. (Caveat from **BDA3 §6.3, pp. 151–153**: posterior-predictive p-values are *not* Uniform under the true model the way classical p-values are. For **many** test statistics they tend to be **conservative** — concentrated near 0.5 — because the same data informs both the model fit *and* the reference predictive distribution it is compared against; this is not universal across all test quantities. Treat $p_B$ as a *flag*, not a hypothesis test, and read the magnitude, not a 0.05 cutoff.)

ArviZ computes and plots these with `az.plot_bpv`:

```python
# Bayesian p-value plot for the whole observation, and for a chosen statistic
az.plot_bpv(idata, kind="p_value")                      # overall calibration
az.plot_bpv(idata, kind="t_stat", t_stat="std")         # is the SD reproduced?
# kind ∈ {"u_value", "p_value", "t_stat"}; t_stat can be "mean","std","median", a quantile, etc.
```

> ⚠️ **Pitfall — use *informative* test statistics.** The single biggest PPC mistake is checking only **location** statistics like the mean. *Almost every model*, even a badly wrong one, reproduces the mean of the data — fitting the mean is the easy part. The mean's Bayesian p-value will be a reassuring ~0.5 on a terrible model. The *informative* statistics are the ones your model could plausibly get wrong: the **minimum**, the **maximum**, the **standard deviation** (catches over/under-dispersion — the classic Poisson-vs-NegativeBinomial tell), the **number of zeros** (catches zero-inflation), the **skew**. Check those. A model that reproduces the max, the sd, and the zero count is a model that has earned some trust.

> 🩺 **Diagnostic — reading `az.plot_bpv`.** With `kind="p_value"`, ArviZ histograms the per-observation Bayesian p-values; under a well-calibrated model they cluster near 0.5 and the plot looks symmetric and centered. With `kind="t_stat"`, you get the predictive distribution of the chosen statistic with the observed value marked and the $p_B$ annotated; you want the observed marker near the middle. `kind="u_value"` plots the PIT values, which under good calibration should be **Uniform** — a ∪ shape means the model is over-confident (predictions too narrow), a ∩ shape means under-confident (too wide). (Confirm the exact `t_stat`/`kind` kwargs against your installed ArviZ — `# verify in current ArviZ docs` — the dossier flags these as version-sensitive.)

### LOO-PIT — the honest PPC that doesn't double-dip

There is a subtle but real problem with everything above: **the data is used twice** — once to fit the model, once to check it. A model can look great on a PPC precisely *because* it was fit to the very data it's being checked against. This double-dipping makes ordinary PPCs *over-optimistic*; they can pass a model that would fail on fresh data.

The fix is **LOO-PIT** (leave-one-out probability integral transform). For each observation $i$, ask: under the model fit to *everything except* $y_i$, where does $y_i$ fall in the predictive distribution? Formally,

$$ u_i = \Pr\!\big(\tilde y_i \le y_i \,\big|\, y_{-i}\big). $$

> 🧮 **The math:** If the model is well-calibrated, these leave-one-out PIT values $u_i$ are **Uniform(0,1)** — each held-out point is, on average, a random quantile of its own out-of-sample predictive distribution. Deviations from uniform diagnose miscalibration, *and because the checked point was held out, there's no double-dipping.* The expensive part — refitting $n$ times — is avoided by computing the leave-one-out predictive cheaply via **PSIS** (Pareto-smoothed importance sampling; the LOO machinery is Chapter 11's subject). You just need a `log_likelihood` group in your InferenceData.

```python
# LOO-PIT needs a log_likelihood group. In recent PyMC v5 it may not be
# computed by default — add it if needed:
with recovery_model:
    pm.compute_log_likelihood(idata)        # writes idata.log_likelihood
    # (alternatively: pm.sample(..., idata_kwargs={"log_likelihood": True}))

az.plot_loo_pit(idata, y="y_obs")           # LOO-PIT KDE vs the uniform band
az.plot_loo_pit(idata, y="y_obs", ecdf=True)  # ECDF-difference version, often clearer
```

> 🩺 **Diagnostic — reading LOO-PIT.** ArviZ overlays the LOO-PIT density (or ECDF) on a grey envelope representing where a truly-uniform sample should lie. **What good looks like:** the LOO-PIT curve stays inside the grey band — flat-ish for the KDE, hugging the diagonal for the ECDF. **The two classic failures, and exactly what they mean:** a **∩ shape** (LOO-PIT piled in the middle) means the predictions are **over-dispersed** — the model is *too uncertain*, its intervals too wide. A **∪ shape** (piled at 0 and 1) means **under-dispersed** — the model is *over-confident*, too many real points land in the tails of its predictive, its intervals are too narrow. A **tilt / monotone trend** means **bias** — the model is systematically too high or too low. This single plot turns "is my model calibrated?" into a shape you can read in one glance, and it is the calibration check you'll reuse in **Chapter 11** alongside LOO model comparison.

> 🔧 **In practice:** Run all three layers. The `plot_ppc` overlay for the global shape, `plot_bpv` with *informative* test statistics for specific features, and `plot_loo_pit` for honest, double-dip-free calibration. They catch different things. A model that passes all three has retrodicted your data convincingly — Betancourt would let you proceed to step 15 (celebrate).

---

## 7. Simulation-based calibration (SBC) — proving the *inference* is correct

We now reach the most rigorous tool in the workflow, and the one most people skip — which is a shame, because it is the only check that validates **the inference algorithm itself**, end to end, across your *whole* prior. It is the statistical big brother of the one-shot parameter recovery from §3. It is Gelman §4.2, Betancourt step 10 (Algorithmic Calibration), and the formal answer to Question Two (computational faithfulness).

> 📜 **Citation/Origin:** Talts, Betancourt, Simpson, Vehtari, Gelman, **"Validating Bayesian Inference Algorithms with Simulation-Based Calibration"** (2018), arXiv:1804.06788. SBC generalizes "simulate-fit-recover" into a *self-consistency theorem* about the entire Bayesian model + algorithm.

### The idea, and why it's exact

Recall the workflow we ran for one-shot recovery: draw a true $\theta^\*$, simulate $y$, fit, check the posterior covers $\theta^\*$. SBC does this **many times** — but instead of asking "is the truth inside the interval?", it asks a sharper question that has an *exact* answer.

Here is the key identity. Suppose you (i) draw $\tilde\theta \sim p(\theta)$ from the **prior**, (ii) draw $\tilde y \sim p(y \mid \tilde\theta)$ from the likelihood, and (iii) draw $L$ posterior samples $\{\theta^{(1)}, \dots, \theta^{(L)}\} \sim p(\theta \mid \tilde y)$. Now compute the **rank** of the true $\tilde\theta$ among those $L$ posterior draws — i.e. how many of the posterior draws are below $\tilde\theta$:

$$ r = \#\{\, \ell : \theta^{(\ell)} < \tilde\theta \,\} \in \{0, 1, \dots, L\}. $$

> 🧮 **The math:** **If your inference is exactly correct, that rank is Uniform on $\{0, 1, \dots, L\}$.** Note the count-below statistic takes **$L+1$ distinct values** $\{0,\dots,L\}$, so when you histogram the ranks, choose a bin count that *divides $L+1$* (or use the ECDF-difference plot) — otherwise a bin-count/rank-range mismatch produces a misleadingly ragged histogram from pure aliasing, not miscalibration. This is not a heuristic — it is a theorem (Talts et al. 2018, building on the fact that the prior is the average of the posterior over the prior predictive: $p(\theta) = \int p(\theta \mid y)\, p(y)\, dy$). Intuitively: when you draw the truth from the prior and then draw posterior samples for data generated by that truth, the truth is *exchangeable* with the posterior draws — it's just another draw from the same distribution — so its rank among them is uniform. Repeat the whole process thousands of times, collect all the ranks, and **histogram them. A correct algorithm gives a flat (uniform) histogram.** Any systematic departure from flat is a *proof* that something — the model code, the sampler, or the prior implementation — is miscalibrated.

This is profound: it converts "is my Bayesian inference correct?" — which sounds unanswerable, because you never know the true posterior — into "is this histogram flat?", which you can *see*.

### Reading the rank histogram — the shape→diagnosis catalog

The shape of the SBC rank histogram is a precise diagnostic. Memorize these four shapes:

```
  UNIFORM (flat)         ∪-SHAPE (U)           ∩-SHAPE (n)          SLOPED / SPIKE
  ▁▁▁▁▁▁▁▁▁▁            █▁▁▁▁▁▁▁▁█            ▁▁▁███▁▁▁            ▁▂▃▄▅▆▇█
  calibrated ✓          posterior TOO         posterior TOO        BIASED
                        NARROW                WIDE                 (systematic
                        (over-confident)      (under-confident)     over/under-estimate)
```

| Rank-histogram shape | What it means | Typical cause / fix |
|---|---|---|
| **Uniform / flat** | Inference is calibrated ✓ | Nothing — proceed |
| **∪ (peaks at both ends)** | Posterior is **too narrow** — over-confident; the truth lands in the tails too often | Under-estimated uncertainty; a sampler not exploring tails; a too-tight or wrong likelihood. Check ESS-tail, priors |
| **∩ (peak in the middle)** | Posterior is **too wide** — under-confident; truth lands in the middle too often | Over-estimated uncertainty; sometimes a too-diffuse prior; occasionally a sign the model is over-parameterized |
| **Sloped / spike at an edge** | **Biased** — posterior systematically shifted away from truth | A bug: sign flip, wrong index, off-by-one, mislabeled parameter, or a genuinely biased approximation (e.g. mean-field VI) |

> 🩺 **Diagnostic:** The ∪/∩ distinction is the one to burn in. **∪ = under-dispersed = over-confident** (the dangerous one — you'll report intervals that are too tight and be wrong more often than you claim). **∩ = over-dispersed = under-confident** (wasteful but safe-ish). A **slope** is a bias bug. Note this is the *same vocabulary* as LOO-PIT (§6) — ∪/∩/tilt — because both are PIT-based calibration checks; LOO-PIT calibrates against held-out data, SBC calibrates the algorithm against the prior. Different targets, same shapes.

> 💡 **Intuition:** SBC is how you catch the bug that passes *every other check.* Your priors look sane (prior predictive passes), your sampler converges ($\hat R$, ESS, divergences all green), your model retrodicts the data (PPC passes) — and yet there's a subtle sign error or a non-centering that biases the posterior in a way no single fit reveals. SBC, running the inference thousands of times against known truths drawn from the whole prior, is the only thing that will catch it. It's expensive, so you reserve it for models you'll *trust with something important* — but for those, it's non-negotiable. This is the heart of Betancourt's "calibrate before you fit real data."

### Running SBC: the `simuk` package

The maintained, PyMC/Bambi-compatible SBC tool is `simuk`. The API is mercifully short:

```python
# pip install simuk   (requires Python >= 3.11)
from simuk import SBC

sbc = SBC(
    recovery_model,                 # your PyMC (or Bambi) model — the real one
    num_simulations=1000,           # >= 1000 recommended for a readable histogram
    sample_kwargs={"draws": 25, "tune": 50, "progressbar": False},
)
sbc.run_simulations()               # this is the expensive part: ~1000 fits
sbc.plot_results()                  # SBC rank histogram / ECDF; want uniform within the band
# verify exact method/attribute names against the installed simuk version
```

`simuk` draws from your model's prior, simulates data, refits, computes ranks for every parameter, and plots the rank histograms (or the ECDF-difference version, which is often easier to read against a confidence envelope). You want every parameter's histogram flat / inside the band.

> 🔧 **In practice:** SBC is *deliberately* run with short chains (`draws=25, tune=50` above) because you're doing it ~1000 times — you need the *rank distribution*, not a precise single posterior, and even short chains give valid ranks as long as they've converged **and the retained draws are roughly independent (thin if autocorrelation is high)**. That independence requirement is not optional: NUTS draws are autocorrelated, and ranking the truth against un-thinned, correlated draws biases the histogram toward the center (a spurious ∩-shape) even when the model is correct — which is exactly why the hand-rolled loop below thins (Talts et al. 2018). And run SBC on the *simulated* ensemble — never the real data — exactly as Betancourt's step 10 prescribes, *before* step 12.

### Hand-rolling the SBC loop (so you understand it)

If `simuk` isn't available, or you just want to see the gears turn, SBC is short to write yourself. This is the whole algorithm — `# pseudocode`-ish but close to runnable:

```python
# Hand-rolled SBC for the linear-regression model. The full algorithm in ~25 lines.
THIN = 5                                       # keep 1 of every 5 draws -> ~independent
L = 100                                        # number of RETAINED (thinned) draws per replication

def sbc_once(N, L=L, thin=THIN, seed=None):
    """One SBC replication: draw truth from prior, simulate, fit, return ranks in {0,...,L}."""
    r = np.random.default_rng(seed)
    # (1) draw a TRUE theta from the PRIOR (must match the model's priors exactly)
    a_t     = r.normal(0, 5)
    b_t     = r.normal(0, 2)
    sigma_t = np.abs(r.normal(0, 2))          # HalfNormal(2)
    # (2) simulate data from the likelihood at that truth
    x_s = r.normal(size=N)
    y_s = r.normal(a_t + b_t * x_s, sigma_t)
    # (3) fit the model to the simulated data; sample L*thin raw draws so that
    #     after thinning we retain exactly L draws to rank against
    with pm.Model() as m:
        a = pm.Normal("a", 0, 5)
        b = pm.Normal("b", 0, 2)
        sigma = pm.HalfNormal("sigma", 2)
        pm.Normal("y", a + b * x_s, sigma, observed=y_s)
        idata = pm.sample((L * thin) // 4, tune=200, chains=4, progressbar=False,
                          random_seed=seed)
    # (4) thin so the L ranks are ~independent (Talts et al. 2018), then rank each
    #     truth among the thinned draws -> rank in {0, 1, ..., L}
    post = idata.posterior
    draws_a     = post["a"].values.ravel()[::thin][:L]
    draws_b     = post["b"].values.ravel()[::thin][:L]
    draws_sigma = post["sigma"].values.ravel()[::thin][:L]
    ranks = {
        "a":     int((draws_a     < a_t).sum()),
        "b":     int((draws_b     < b_t).sum()),
        "sigma": int((draws_sigma < sigma_t).sum()),
    }
    return ranks

# (5) repeat many times, collect ranks, histogram each parameter -> want UNIFORM.
# Ranks live in {0,...,L}, i.e. L+1 possible values, so use a bin count that divides
# L+1 (here L=100 -> 101 values; bins=101 or any divisor avoids aliasing artifacts).
all_ranks = [sbc_once(N=100, seed=s) for s in range(1000)]   # slow; this is the cost
# then: plt.hist([d["b"] for d in all_ranks], bins=L + 1)  -> should look flat
```

Walk the five steps once more, because they *are* SBC: (1) the truth comes from the **prior**, not an arbitrary value — that's what makes the rank-uniformity theorem hold; (2) simulate data at that truth; (3) fit *your real model*; (4) compute the rank of the truth among posterior draws; (5) aggregate over many replications and check the histogram is flat. If `b`'s histogram came out ∪-shaped, you'd know your posterior for `b` is too narrow — over-confident — *before* a single real number was ever reported.

> ⚠️ **Pitfall — the prior in your SBC simulator must match the prior in your model, exactly.** This is the most common SBC bug. If you simulate truths from `Normal(0, 5)` but the model declares `Normal(0, 3)`, the rank-uniformity theorem doesn't apply and you'll see a spurious non-uniform histogram that diagnoses a "problem" that is really just your mismatched simulator. Generate the truth from the *literal same* prior the model uses.

---

## 8. The discipline of building up — expand only when a check demands it

Now the philosophy that ties the tools together. You have a loop and four checks. The remaining question is: **when do you make the model bigger?** The answer, and it is the whole craft, is:

> **Start with the simplest model that could plausibly work. Add complexity only when a specific check demands it — and let the check tell you exactly what to add.**

This is Gelman's "start simple *and* complex, meet in the middle" (§5.2) and the "topology of models" (§7.4), and it is the moral of the golf-putting story (§10). It is also just disciplined engineering. Here is why each direction of the rule matters:

**Why start simple.** A simple model is *debuggable*. When a one-parameter model fails a check, you know precisely where to look. When a thirty-parameter hierarchical-spline-mixture fails, you're hunting in the dark. Every piece of complexity you add is a place a bug can hide and a place the geometry can go bad (folk theorem). So you earn complexity; you don't assume it.

**Why expand when a check demands it.** A model that passes every check *is allowed to stay simple* — adding more would be overfitting and self-indulgence. But when a check fails, it doesn't just say "wrong," it says **how**: a PPC that misses the data's extra zeros says "add zero-inflation"; a PPC where the tails are too thin says "go from Normal to Student-$t$"; a Bayesian p-value on the SD near 1 says "your counts are overdispersed — Poisson → Negative Binomial"; a forest plot showing every group estimate pulled to a common value with no room to vary says "let the slopes vary too." **The failed check is a pointer to the missing model component.** That is the engine of the whole iterative loop.

> 💡 **Intuition — the owl.** McElreath jokes about textbooks that "draw the owl" in two steps: (1) draw two circles, (2) draw the rest of the owl. The workflow is the antidote — it draws the owl one *checkable* line at a time. Each model in the sequence is a complete, fitted, checked owl; the next one fixes the specific feather the last check flagged. You are never staring at a blank page wondering how to build the final model; you're always just fixing the one thing your last check complained about.

Here is the expansion loop as a concrete table you can act on:

| A check failed like this... | ...so the model is missing... | ...expand toward |
|---|---|---|
| PPC: observed has more zeros than predictive | excess zeros | zero-inflated / hurdle (Ch 13) |
| `plot_bpv` on SD: $p_B \approx 0$ for counts | overdispersion | Poisson → Negative Binomial (Ch 09) |
| PPC: observed tails fatter than predictive | too-thin error tails | Normal → Student-$t$ (robust regression, Ch 09) |
| LOO-PIT ∪-shape | over-confidence | check priors / add a dispersion parameter |
| Forest plot: groups differ but model pools them | group structure | varying intercepts/slopes (hierarchical, Ch 10) |
| Residuals show curvature vs a predictor | nonlinearity | splines / GP / interaction (Ch 12, 14) |
| Divergences won't clear | bad geometry | reparameterize: non-centered (Ch 07, 10) |

> 🔧 **In practice:** Keep every model in the sequence. Don't overwrite `model_v1` with `model_v2`. You'll want to *compare* them — with LOO/`az.compare` (Chapter 11) — to confirm that each expansion actually bought predictive improvement and wasn't just extra parameters chasing noise. The workflow doesn't end at "a model that passes checks"; it ends at "the simplest model in my sequence that passes checks *and* isn't beaten on out-of-sample prediction by a simpler sibling." That comparison step is **Chapter 11's** whole subject.

> ⚠️ **Pitfall — expanding without a reason.** The mirror-image error of starting too complex is expanding *speculatively* — "let me just add a varying slope and an interaction and a spline, to be safe." Every unmotivated piece is a place for the sampler to struggle and a parameter fit to noise. If no check is asking for a component, don't add it. Complexity is a liability you take on only to fix a documented failure.

---

## 9. Worked end-to-end: the full mini-workflow on the Howell !Kung heights

Time to do the whole thing, visibly, on a real dataset — every stage of the spine, in order, on data we did not generate. We'll use the **Howell !Kung** dataset (the course's flagship for first models, introduced in Ch 01/03): partial census data from Nancy Howell's 1960s fieldwork with the !Kung San. We'll model adult **height as a function of weight** — a genuinely linear relationship for adults, which makes it perfect for watching the workflow run clean and then *deliberately break and expand* a piece.

### Step 0 — Domain expertise & the question (Conceptual Analysis)

Before any code: what do we know? Adult human heights are positive, roughly 130–190 cm, and taller people tend to weigh more, plausibly linearly over the adult range. Weight is positive, tens of kg. The question: *how does expected height change per kg of weight, with honest uncertainty?* This is Betancourt step 1 and Gelman §2.1 — and it already tells us the rough scale every prior must respect.

### Step 1 — Load and look at the data

```python
url = "https://raw.githubusercontent.com/rmcelreath/rethinking/master/data/Howell1.csv"
d = pd.read_csv(url, sep=";")            # rethinking CSVs are semicolon-separated
adults = d[d.age >= 18].copy()           # the height~weight line is linear only for adults
print(adults[["height", "weight"]].describe())
# height ~ 154 cm mean (≈135–180 range); weight ~ 45 kg mean
```

### Step 2 — Standardize, then build the generative model

Per the course default (Ch 03), we **standardize the predictor** so a unit-scale slope prior is meaningful and the sampler sees a well-conditioned geometry:

```python
adults["weight_z"] = (adults.weight - adults.weight.mean()) / adults.weight.std()
```

Now the generative model. Thinking generatively (Ch 01): each person's height is Normal around a line in standardized weight.

$$ h_i \sim \mathrm{Normal}(\mu_i, \sigma), \qquad \mu_i = \alpha + \beta\, z_i, $$
$$ \alpha \sim \mathrm{Normal}(155, 20), \quad \beta \sim \mathrm{Normal}(0, 10), \quad \sigma \sim \mathrm{HalfNormal}(20). $$

The intercept prior is centered at a plausible mean adult height (155 cm) with room to be wrong; the slope is on the *standardized* scale (so $\beta$ is "cm per standard deviation of weight"), weakly informative; $\sigma$ is a positive scale with a half-Normal (the course-default scale prior, Ch 02–03).

```python
coords = {"obs_id": adults.index}
with pm.Model(coords=coords) as howell_v1:
    z = pm.Data("weight_z", adults.weight_z.values, dims="obs_id")
    alpha = pm.Normal("alpha", mu=155, sigma=20)
    beta  = pm.Normal("beta",  mu=0,   sigma=10)
    sigma = pm.HalfNormal("sigma", sigma=20)
    mu = pm.Deterministic("mu", alpha + beta * z, dims="obs_id")
    height = pm.Normal("height", mu=mu, sigma=sigma,
                       observed=adults.height.values, dims="obs_id")
```

### Step 3 — Prior predictive check (gate 1, *before* fitting)

```python
with howell_v1:
    idata_v1 = pm.sample_prior_predictive(draws=200, random_seed=RANDOM_SEED)
az.plot_ppc(idata_v1, group="prior", num_pp_samples=200)

pp = idata_v1.prior_predictive["height"].values
print(f"prior-predictive height range: [{pp.min():.0f}, {pp.max():.0f}] cm")
```

> 🩺 **Reading it:** The prior-predictive heights should span a believable-but-generous range — say roughly 80–230 cm — wide enough to be honest, not so wide it predicts negative or 4-metre humans. If you'd set `beta ~ Normal(0, 100)`, the implied heights would swing wildly with weight and the cloud would be absurd; the check would catch it and you'd tighten `beta`. Our priors pass: the implied datasets look like plausible human heights. (This is exactly the Ch 03 prior-predictive workflow, now as a *gate*.)

### Step 4 — Fit, then the computational-diagnostics gate (gate 2)

```python
with howell_v1:
    # idata_v1 already holds the prior / prior_predictive groups from Step 3.
    # pm.sample() returns a FRESH InferenceData (posterior + sample_stats); .extend()
    # MERGES those new groups into idata_v1, so prior, posterior, and sample_stats
    # all live in one object. (Earlier sections built a flat `idata = pm.sample(...)`
    # because nothing preceded it; here we extend because the prior groups came first.)
    idata_v1.extend(pm.sample(1000, tune=1000, chains=4, target_accept=0.9,
                              random_seed=RANDOM_SEED))

print("divergences:", int(idata_v1.sample_stats["diverging"].sum()))   # expect 0
az.summary(idata_v1, var_names=["alpha", "beta", "sigma"])
az.plot_trace(idata_v1, var_names=["alpha", "beta", "sigma"])
```

> 🔧 **In practice:** `.extend()` is how you keep a *single* InferenceData that accumulates every group as the workflow runs — prior/prior_predictive (Step 3), then posterior/sample_stats (here), then posterior_predictive and log_likelihood (Step 5). The flat `idata = pm.sample(...)` pattern in §3–§6 and the `.extend()` pattern here are interchangeable; the only rule is that each group name appears once, so you `.extend()` (or pass `extend_inferencedata=True`) rather than reassigning when an object already holds groups you want to keep.

Expected `az.summary` (your digits will be close):

| | mean | sd | hdi_3% | hdi_97% | ess_bulk | ess_tail | r_hat |
|---|---|---|---|---|---|---|---|
| alpha | 154.6 | 0.27 | 154.1 | 155.1 | 4200 | 3100 | 1.00 |
| beta | 6.0 | 0.27 | 5.5 | 6.5 | 4300 | 3000 | 1.00 |
| sigma | 5.1 | 0.19 | 4.7 | 5.4 | 4000 | 2900 | 1.00 |

> 🩺 **Reading the gate:** **0 divergences**, all $\hat R = 1.00 \le 1.01$, ESS in the thousands ($\gg 400$). The trace plots are fuzzy caterpillars with the four chains overlapping. This fit passes the gate — we are *now allowed* to interpret it. Substantively: `beta ≈ 6.0` means each standard deviation of weight (~6.2 kg) buys about 6 cm of expected height, and `alpha ≈ 154.6` cm is the expected height at average weight (because we standardized, the intercept is the mean-weight prediction — a nice side effect of standardization).

### Step 5 — Posterior predictive / retrodictive check (gate 3)

```python
with howell_v1:
    pm.sample_posterior_predictive(idata_v1, extend_inferencedata=True,
                                   random_seed=RANDOM_SEED)
az.plot_ppc(idata_v1, num_pp_samples=100)
az.plot_bpv(idata_v1, kind="t_stat", t_stat="std")   # is the spread reproduced?
```

> 🩺 **Reading it:** The observed height KDE should thread the middle of the posterior-predictive cloud — the model generates height distributions like the real one. The Bayesian p-value on the standard deviation should sit near 0.5 (the model reproduces the spread). For this well-matched linear model on adults, the retrodictive check passes cleanly. Add a `log_likelihood` group and the LOO-PIT for the honest, non-double-dipping version:

```python
with howell_v1:
    pm.compute_log_likelihood(idata_v1)
az.plot_loo_pit(idata_v1, y="height", ecdf=True)   # want it inside the uniform band
```

A LOO-PIT curve hugging the diagonal (inside the grey envelope) confirms calibration: no ∪ (over-confidence), no ∩ (under-confidence), no tilt (bias).

### Step 6 — See the loop fire: break a check, let it tell you what to add

The workflow's power is in *failure*, so let's manufacture one. Suppose we'd been sloppy and included **children** (whose height-weight relationship is sharply nonlinear) but kept the straight-line model:

```python
allages = d.copy()
allages["weight_z"] = (allages.weight - allages.weight.mean()) / allages.weight.std()
# ... refit howell_v1's straight-line model to ALL ages ...
# PPC now FAILS: az.plot_ppc shows the observed (bimodal-ish, kids + adults) density
# walking outside the unimodal predictive cloud; residuals curve systematically vs weight.
```

> 🩺 **Reading the failure → the fix:** The PPC overlay shows the observed density spilling outside the predictive cloud, and a plot of residuals against weight shows a clear *curve*, not noise. The check isn't just saying "wrong" — it's pointing at **nonlinearity**. Per the §8 expansion table, the model is missing a way to bend: the fix is a **spline or a polynomial in weight, or a Gaussian process** (Chapters 12 and 14). We'd add exactly that one component, refit, and re-run all three gates — and *only* that component, because that's the only thing the check demanded. That is the iterative loop, observed in the wild: a failed posterior predictive check naming the missing piece, an expansion, and a re-check.

### Step 7 — (For a model you'll trust) calibrate with SBC

For our clean adults-only `howell_v1`, before betting anything important on `beta`, we'd run SBC over its prior (§7) — `simuk.SBC(howell_v1, num_simulations=1000, ...)` — and confirm flat rank histograms for `alpha`, `beta`, `sigma`. Flat histograms certify that the NUTS inference recovers truth across the whole prior, not just for this one dataset. That's the difference between "it fit" and "I've *proven* it fits."

### Step 8 — Summarize and propagate uncertainty (use the model)

With all gates green, we finally *use* the posterior — and crucially, we propagate full uncertainty, not a point estimate (Gelman §6.4, Betancourt's "celebrate"). For a prediction at a new weight, swap the data and sample the posterior predictive:

```python
new_weight_z = (np.array([50.0]) - adults.weight.mean()) / adults.weight.std()
with howell_v1:
    # swap the data AND its coordinate (the named dim changes length: N -> 1)
    pm.set_data({"weight_z": new_weight_z}, coords={"obs_id": [0]})
    pred = pm.sample_posterior_predictive(
        idata_v1, var_names=["height"], predictions=True,
        extend_inferencedata=False, random_seed=RANDOM_SEED)
# pred.predictions["height"] is a full predictive DISTRIBUTION for a 50 kg adult:
# report a 94% interval, not a single number.
az.plot_forest(idata_v1, var_names=["beta"], combined=True)   # the effect, with uncertainty
```

The deliverable is never "the slope is 6.0." It's "the slope is 6.0 cm per SD of weight, 94% credible interval [5.5, 6.5], and a 50 kg adult is predicted to be about X cm tall with a 94% predictive interval of [lo, hi]" — the full posterior, honestly propagated. That is what the whole workflow was for.

---

## 10. Modeling as software development (Gelman §9)

One stage of Gelman's workflow has nothing to do with statistics and everything to do with not shipping garbage: **treat your modeling as software development.** The bugs that the four statistical checks catch are real, but so are the mundane ones — a swapped variable, an un-pinned dependency, a result you can't reproduce next week. The professional habits:

- **Version control your models.** Keep `model_v1`, `model_v2`, … as separate, committed files. The sequence *is* your analysis; the diffs between models tell the story of what each check forced you to add.
- **Set and pass a seed everywhere.** `RANDOM_SEED = 8927` at the top, passed to `pm.sample`, `pm.sample_prior_predictive`, `pm.sample_posterior_predictive`, and your simulators. Bayesian results are stochastic; reproducibility is non-optional.
- **Pin your stack.** This course is frozen to **PyMC v5** (`"pymc>=5.16,<6"`) — the docs now default to v6, which would silently change behavior. Pin it, and check `pm.__version__` prints `5.x`.
- **Turn recovery and SBC into tests.** The automated coverage check from §3 belongs in a test suite as an `assert`. A model with a passing parameter-recovery test is a model you can refactor without fear.
- **Keep the InferenceData.** Save `idata` to disk (`az.to_netcdf`); it carries the posterior, sample stats, prior, predictive, and observed data together, so your diagnostics are reproducible from the artifact without re-sampling.

> 💡 **Intuition:** Everything in this chapter is a defense against *fooling yourself*. The statistical checks defend against fooling yourself about the model; the software hygiene defends against fooling yourself about *what you actually ran*. You need both. A brilliant model checked into no version control with no seed is an anecdote, not an analysis.

---

## ⚠️ Common errors & how to fix them

Workflow-level traps — the ones that come from misusing or skipping a *stage*. (Sampler/geometry-specific errors live in Chapter 07's table; prior-design errors in Chapter 03's.)

| Symptom | Cause | Fix |
|---|---|---|
| Posterior looks fine but is subtly wrong; every diagnostic is green | Skipped fake-data recovery / SBC; a sign-flip or index bug hides behind plausible numbers | Always run §3 parameter recovery before real data; run §7 SBC for models you'll trust. Green $\hat R$ ≠ correct model |
| `sample_prior_predictive(samples=...)` errors | v5 uses `draws=`, not `samples=` | Use `draws=`; results land in `idata.prior` / `idata.prior_predictive` |
| Prior predictive draws are physically absurd (heights of $10^6$ cm) | Vague priors × unstandardized predictors → small-world nonsense | **Standardize predictors** (Ch 03), weakly-informative priors on a sensible scale, re-check prior predictive until data looks like data |
| `az.plot_ppc` raises "no `posterior_predictive` group" | Forgot to draw posterior predictive samples | `pm.sample_posterior_predictive(idata, extend_inferencedata=True)` first |
| PPC overlay looks perfect, yet the model is wrong | **Double-dipping** (data used to fit *and* check) + only checked location stats | Use **LOO-PIT** (`az.plot_loo_pit`) and **informative** test stats (min/max/sd/#zeros), not the mean |
| `az.plot_loo_pit` / `az.loo_pit` errors on missing log-likelihood | `log_likelihood` group not computed (recent v5 default) | `pm.compute_log_likelihood(idata)` or `pm.sample(..., idata_kwargs={"log_likelihood": True})` |
| Bayesian p-value for the **mean** is ~0.5 so "the model is fine" | The mean is the *easy* statistic — almost every model reproduces it | Judge the model on **dispersion, extremes, zeros, skew**, not location |
| SBC histogram non-uniform but the model is actually fine | Simulator's prior ≠ model's prior; or too few simulations / autocorrelated draws | Draw truths from the **exact** model priors; use `num_simulations ≥ 1000`; thin the posterior draws |
| Chasing divergences with `target_accept=0.99` forever | **Folk theorem ignored** — it's a model problem, not a tuning problem | After 1–2 tuning bumps, *change the model*: reparameterize (non-centered, Ch 07/10), tighten scale priors, marginalize |
| Jumped straight to a huge model that won't fit | Started complex instead of building up | Start with the simplest plausible model; expand only when a check demands it (§8); keep every version for comparison (Ch 11) |
| Results don't reproduce next week | No seed / un-pinned stack / overwritten model file | Seed everything, pin PyMC to v5, version-control each model, save `idata` with `az.to_netcdf` |

---

## 🧪 Exercises

**1. (Conceptual — map the spine.)** Without looking back, draw the six-box workflow spine from memory, and against each box write (a) the Gelman §-number and (b) the Betancourt step number(s) that correspond. Then write one sentence on the *temperamental* difference between the two authors (descriptive/iterative vs prescriptive/calibrate-first). *Hint: the reconciliation table in §2C is the answer key — close it first.*

**2. (Coding — parameter recovery on a model with a hidden bug.)** Take the §3 linear-regression recovery code and deliberately introduce a bug: in the model, write `mu = a - b * x_data` (a sign flip on the slope) while still generating data with `+`. Fit and run the automated coverage check. Which parameter's truth lands outside its HDI, and why does the sign flip *not* corrupt `a` or `sigma`? Now fix the bug and confirm recovery. *Hint: a sign flip maps `b → -b`; recovery will cover `-b_true`, not `b_true`.*

**3. (Coding — make a PPC fail, then read the diagnosis.)** Simulate overdispersed count data: `y = rng.negative_binomial(n=5, p=0.3, size=200)`. Fit a **Poisson** model to it (the wrong likelihood). Run `az.plot_ppc` and `az.plot_bpv(idata, kind="t_stat", t_stat="std")`. Describe exactly how the PPC reveals the misfit and what the Bayesian p-value on the SD is near. Then refit with `pm.NegativeBinomial` and show the check passes. *Hint: Poisson forces mean = variance; the SD's Bayesian p-value should be near 0 for the Poisson fit.*

**4. (Coding — LOO-PIT vs a naive PPC.)** Fit a Normal model to clearly heavy-tailed data (`y = rng.standard_t(df=3, size=300) * 3`). Show that the ordinary `az.plot_ppc` overlay looks *almost* fine (the bulk matches) but `az.plot_loo_pit` reveals a ∪ shape. Explain in two sentences why LOO-PIT catches the over-confidence that the overlay nearly hides, and what model expansion the ∪ shape points to. *Hint: ∪ = under-dispersed = over-confident; the fix is Normal → Student-$t$.*

**5. (Coding — hand-rolled SBC.)** Run the §7 hand-rolled `sbc_once` loop for the *correct* linear model with `num_simulations=200` (use short chains; it's slow), and histogram the ranks for `b`. Is it flat? Now corrupt it: make the model's `sigma` prior `HalfNormal(0.5)` while the simulator still draws `sigma` from `HalfNormal(2)`. Re-run and describe the rank histogram's shape. Which §7 failure mode does it match, and why is the *mismatch itself* (not a real bug) the cause? *Hint: the simulator-prior ≠ model-prior pitfall.*

**6. (Conceptual + judgment — the expansion decision.)** You fit a hierarchical model; the diagnostics gate is clean, the PPC overlay passes, but a forest plot shows all group intercepts pulled to nearly the same value with tight intervals. A colleague says "add varying slopes *and* an AR(1) term *and* a spline, to be safe." Using §8, argue what (if anything) you should add and in what order, and how you'd *confirm* each addition earned its place. *Hint: invoke "expand only when a check demands it" and Chapter 11's LOO comparison.*

---

## 📚 Resources & further reading

The two workflow papers first — read them in full at least once; they are the source of everything in this chapter.

- **Gelman, Vehtari, Simpson, Margossian, Carpenter, Yao, Kennedy, Gabry, Bürkner, Modrák, "Bayesian Workflow" (2020)** — *the* workflow paper. The §2–§8 map in this chapter is its table of contents; §5.1 is the folk theorem; §10 (golf putting) is the canonical model-expansion narrative. https://arxiv.org/abs/2011.01808
- **Betancourt, "Towards A Principled Bayesian Workflow"** — the four questions, fifteen steps, three phases; rigorous and prescriptive; the source of "retrodictive checks" and "calibrate before you fit real data." https://betanalpha.github.io/assets/case_studies/principled_bayesian_workflow.html · source `.Rmd` (exact step titles): https://github.com/betanalpha/knitr_case_studies/tree/master/principled_bayesian_workflow
- **Betancourt writing index** — gateway to every related case study (HMC, hierarchical, identifiability, robust workflow with PyStan). https://betanalpha.github.io/writing/
- **Talts, Betancourt, Simpson, Vehtari, Gelman, "Validating Bayesian Inference Algorithms with Simulation-Based Calibration" (2018)** — the SBC algorithm (Algorithm 1), the rank-uniformity theorem, and the ∪/∩/slope shape catalog. https://arxiv.org/abs/1804.06788
- **Gabry, Simpson, Vehtari, Betancourt, Gelman, "Visualization in Bayesian workflow" (2019)**, *JRSS-A* 182(2):389–402 — the origin of the PPC, LOO-PIT, and energy-plot visual idioms ArviZ implements. https://doi.org/10.1111/rssa.12378 (code: https://github.com/jgabry/bayes-vis-paper)
- **McElreath, *Statistical Rethinking* 2e** — Ch 1 (Golem of Prague), Ch 2 (Small Worlds & Large Worlds) for the small-world/generative framing that motivates the whole loop; the "draw the owl" gag. Book + free lectures. https://xcelab.net/rm/ · lectures: https://github.com/rmcelreath/stat_rethinking_2023
- **Martin, Kumar, Lao, *Bayesian Modeling and Computation in Python* (2021)** — Ch 2 "Exploratory Analysis of Bayesian Models" (PPC, `az.plot_bpv`, LOO-PIT with ArviZ code) and Ch 9 (end-to-end workflow) parallel this chapter directly. Free online. https://bayesiancomputationbook.com — Ch 2: https://bayesiancomputationbook.com/notebooks/chp_02.html
- **Gelman, Hill, Vehtari, *Regression and Other Stories* (2020, free PDF)** — Ch 9 "Prediction and Bayesian inference" and its workflow framing; excellent companion to §9 here. https://avehtari.github.io/ROS-Examples/
- **BDA3 (Gelman et al. 2013, free PDF)** — Ch 6 "Model checking" is the reference for posterior predictive checks and Bayesian p-values (pp. 143–153). http://www.stat.columbia.edu/~gelman/book/
- **PyMC "Prior and Posterior Predictive Checks" core notebook** — verified v5 idioms for `sample_prior_predictive(draws=...)`, `sample_posterior_predictive(..., extend_inferencedata=True, predictions=...)`, and the group names. https://www.pymc.io/projects/docs/en/latest/learn/core_notebooks/posterior_predictive.html
- **EABM (ArviZ) — "Prior and Posterior predictive checks"** — a clean modern ArviZ treatment of PPC, LOO-PIT, and Bayesian p-values. https://arviz-devs.github.io/EABM/Chapters/Prior_posterior_predictive_checks.html
- **ArviZ docs** — `az.plot_ppc`: https://python.arviz.org/en/v0.21.0/api/generated/arviz.plot_ppc.html · `az.plot_loo_pit`: https://python.arviz.org/en/v0.19.0/api/generated/arviz.plot_loo_pit.html · `az.plot_bpv` (kinds: `u_value`/`p_value`/`t_stat`): https://python.arviz.org/
- **`simuk` (PyPI) — SBC for PyMC/Bambi** — the `SBC(model, num_simulations, sample_kwargs).run_simulations().plot_results()` API used in §7. https://pypi.org/project/simuk/
- **Betancourt, "Robust Statistical Workflow with PyStan"** — the same workflow with Stan idioms and the `check_*` diagnostic functions; useful contrast to the PyMC route. https://betanalpha.github.io/assets/case_studies/pystan_workflow.html

---

## ➡️ What's next

You now have the *method*: a loop, two canonical formulations reconciled to one spine, and the four checks — fake-data recovery, prior predictive, posterior predictive (with Bayesian p-values and LOO-PIT), and SBC — that hold it together, plus the discipline of building up and expanding only on demand. Everything from here on is that loop applied to richer models. **Chapter 09 — GLMs & Bambi** is where the loop gets its first real workout: linear, logistic, Poisson, and Negative-Binomial regressions, robust Student-$t$ models, link functions and how to interpret their coefficients (log-odds, incidence-rate ratios), and the Bambi formula API shown alongside its raw-PyMC equivalent. Every model there is born from a generative story, gated through a prior predictive check, fit, diagnosed, and retrodicted exactly as you just learned — and when a Poisson model's posterior predictive check flags overdispersion, you'll know precisely what the check is telling you to do next, because you met that exact failure-to-expansion move here. The workflow is the engine; the GLM zoo is the first thing you'll drive with it.


