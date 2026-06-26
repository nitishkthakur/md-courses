# Chapter 00 — Welcome & Setup

### The front door to a practice-first course on Bayesian modeling — what it is, how to study it, how to set up your machine, and the workflow that ties everything together.

> *A note from me to you.*
>
> *Welcome. If you've found your way here, you're already a capable ML practitioner — you can fit models, tune them, and ship them. So why learn Bayes? Because at some point a point estimate stops being enough. A model says "churn probability 0.71" and the only honest answer to "how sure are you?" is a shrug. You have eight hospitals, three with twelve patients each, and pooling them all together or treating them as totally separate both feel wrong. You want to inject what you already know — that a coefficient can't be a billion, that an effect is probably small — and have the math respect it. Bayesian modeling gives you a single, coherent language for all of this: a full distribution over everything you don't know, that you can interrogate, check, and propagate into decisions.*
>
> *Here is the promise of this course, and the one mistake I most want to save you from: the mistake of trusting a posterior you never checked. Most disasters in applied Bayesian work are not exotic math failures — they are a vague prior that implied human beings are nine kilometers tall, a sampler that quietly diverged while you looked at the mean, or a model that fit beautifully and predicted nonsense. We are going to build the reflex of never believing a number until the machine has earned your trust. That reflex is the workflow, and this chapter is where you meet it.*
>
> *Read this whole chapter before you write a line of model code in Chapter 01. It sets up your environment, gives you a mental map of the next eighteen chapters, and — most importantly — teaches you how to study the rest. Let's open the door.*

---

## What you'll be able to do after this chapter

- Explain what this course is, who it's for, and why it is **practice-first** rather than proof-first.
- Adopt a concrete **study habit**: for every model you meet, answer the same six questions — generative story, priors, likelihood, how to fit, how to diagnose, and when it breaks.
- Stand up a **clean, reproducible environment** with PyMC v5, ArviZ, Bambi, NumPyro/JAX, nutpie, BlackJAX, PyMC-BART, and CmdStanPy — and run a 5-line "hello Bayes" smoke test that samples and prints a summary.
- Recite the **Bayesian workflow at a glance** as a loop: domain knowledge → generative model → prior predictive check → fit → diagnostics → posterior predictive check → expand/compare → simulation-based calibration.
- Find your way around the **master dataset index** and load any of our shared datasets in one line.
- Read the **notation and conventions** the whole course uses, so nothing surprises you later.
- Choose a **path through Chapters 01–18 + appendix** that fits your time and goals.

---

## 1. What this course is (and what it is not)

This is a complete, expert-level, **practice-first** course on Bayesian modeling and the **Bayesian workflow** in Python. The target reader is you: someone who already knows machine learning and wants to become a genuine Bayesian expert — fluent in both the theory *and* the hands-on craft. By the end you will be able to build, fit, diagnose, debug, compare, and ship Bayesian models across the full zoo: generalized linear models, hierarchical/multilevel models, Gaussian processes, mixtures, nonparametric and BART models, time series, and missing-data models. You'll choose priors deliberately instead of by default, and you'll read every diagnostic instead of squinting at it.

> 💡 **Intuition: "practice-first" is not "theory-free."** We do not skip the math — Bayes' theorem, the ELBO, the geometry of Hamiltonian Monte Carlo, the rank-normalized $\hat{R}$ — all of it is here, derived and explained. But the *organizing principle* is the practitioner's question, "how do I actually build and trust this model?", not the theorem's question, "what is the most general statement I can prove?" Every concept arrives attached to runnable code and a diagnostic you can read.

Think of this folder of chapters as a textbook you can execute. Each chapter is a Markdown file with prose, math (rendered with LaTeX), and PyMC v5 code blocks you are meant to copy, run, and *break*. The course is opinionated about tools — we'll get to exactly which libraries and why in §3 — because consistency is what lets a course this large feel like one coherent voice instead of eighteen disconnected tutorials.

**What this course is not.** It is not a pure-theory monograph (read BDA3 for that — and we'll point you there). It is not a tour of one library's API (the API is a means, not the end). And it is not a sales pitch: Bayesian methods buy you coherence and honest uncertainty at the real price of more computation and more discipline. We will teach the trade-off, not pretend it doesn't exist.

> 🔧 **In practice:** four teachers stand behind this course's voice, and it helps to know whose shoulder you're hearing in any given paragraph. **Richard McElreath** (*Statistical Rethinking*) for intuition-first thinking — golems, small worlds, generative stories, relentless plots. **Michael Betancourt** for rigor — the geometry of the typical set, why HMC works, divergences as truth-tellers, the *principled* workflow. **Osvaldo Martin, Ravin Kumar, and Junpeng Lao** (the *Bayesian Modeling and Computation in Python* and ArviZ crowd) for clean, idiomatic PyMC/ArviZ and production judgment. When you see a vivid metaphor, that's McElreath; when you see a careful caveat about geometry, that's Betancourt; when you see a tidy `az.summary` call, that's Martin and Kumar.

---

## 2. How to study this course

You can read a chapter passively and feel like you understood it. You did not. Bayesian modeling is a *motor skill* — like playing an instrument, it lives in your fingers, not just your head. Here is how to actually learn it.

### 2.1 Run everything. Then break it.

Type the code. Don't copy-paste the first time — *type* it, so your hands learn the idioms. Then change something: widen a prior and watch the prior predictive explode; shrink the data and watch the posterior widen; remove the non-centered reparameterization and watch the divergences appear. The golem only teaches you when you watch it move. Every chapter is designed so the plots and diagnostics are *yours* to generate — when a chapter says "the trace plot will show four fuzzy caterpillars overlapping," you should run it and confirm that's what you see.

### 2.2 The six-question habit (the single most important thing in this chapter)

For **every model** you meet — in this course or in your own work — train yourself to answer the same six questions, in order. This is the spine of expertise. Most practitioners can answer one or two of these for a given model; experts answer all six without being asked.

| # | Question | What you're really asking |
|---|---|---|
| 1 | **What's the generative story?** | If I ran this model *forward*, how would fake data come out? What's the recipe from parameters → data? |
| 2 | **What are the priors, and why?** | What do I believe about each parameter *before* the data, and is that belief on a sane scale? |
| 3 | **What's the likelihood?** | Given parameters, what distribution generated each observation? (Normal? Bernoulli? Poisson? Student-t?) |
| 4 | **How do I fit it?** | Which inference engine — conjugate/grid, quadratic approximation, MCMC/NUTS, variational? With what settings? |
| 5 | **How do I diagnose it?** | $\hat{R}$, ESS, divergences, energy, posterior predictive checks — did the machine actually work, and does the fit reproduce the data? |
| 6 | **When does it break, and which dataset proves it?** | What's the failure mode (a funnel, label switching, an unidentified parameter), and the concrete example that exposes it? |

> 💡 **Intuition:** notice that questions 1–3 *define* the model, question 4 *fits* it, and questions 5–6 *check* it. That ordering — define, fit, check — is the entire Bayesian workflow in miniature. If you internalize the six-question habit, the workflow in §5 will feel like an old friend rather than a new checklist.

I will model this habit for you. Many chapters explicitly answer these six questions for their flagship model, and the appendix collects them into cheat-sheets. When you do your own modeling later, keep a literal checklist of these six until they become reflex.

### 2.3 Internalize the spine, not the fifteen steps

In Chapter 08 you'll meet two formal workflows: Gelman et al.'s descriptive toolbox and Betancourt's prescriptive fifteen-step checklist. Both are excellent and both can feel like a lot. The trick is to memorize only the **spine** — *generative model → prior predictive check → fit → diagnose → posterior predictive check → expand* — and treat everything else as two zoom levels on that same skeleton. We'll preview the spine in §5 of this very chapter.

### 2.4 Always check before you trust

This is the prime directive. No posterior is believed until three things are true: the **prior predictive** looks physically plausible, the **diagnostics are clean** ($\hat{R} \le 1.01$, zero divergences, healthy effective sample size — all defined precisely in Chapter 07), and the **posterior predictive check** shows the fitted model reproducing the salient features of the real data. Skipping any of these is how you end up confidently wrong.

### 2.5 Think generatively, first

For every problem, before you reach for a model, ask McElreath's question: *"How could this data have arisen?"* Write the simulator first — even just in your head, even just `rng.normal(a + b*x, sigma)`. The model almost always falls out of the simulator, because a Bayesian model *is* a simulator with the arrow of inference reversed. We will lean on synthetic-data-with-known-truth constantly, precisely because when you control the generative process you can check whether your inference *recovers* the parameters you put in. That's the cheapest, most honest sanity check there is.

### 2.6 Be honest about costs

Bayesian methods are slower than a `model.fit()` one-liner. MCMC can fail in ways gradient descent does not. Prior elicitation takes thought. And you genuinely have to check your model rather than trusting a single accuracy number. I'll never hide these costs from you. The payoff — coherent uncertainty, principled regularization, a uniform language for hierarchies and latent variables and missing data — is worth it, but you should know what you're paying.

---

## 3. Environment setup (do this now)

Let's get you a working machine. I'll give you a recommended path (conda-forge), a pip-only fallback, the optional bits, and a 5-line smoke test that proves it all works. Do this before Chapter 01.

### 3.1 The stack and why each piece is here

| Library | Import | Role in this course |
|---|---|---|
| **PyMC v5** | `import pymc as pm` | The primary modeling library. Almost every model is written in PyMC. |
| **ArviZ** | `import arviz as az` | *All* diagnostics, summaries, and plots. Inference results live in `az.InferenceData`. |
| **Bambi** | `import bambi as bmb` | Formula-style GLMs/multilevel models. We always show the raw-PyMC equivalent too. |
| **NumPyro / JAX** | `import numpyro` | A fast JAX sampler backend (`pm.sample(nuts_sampler="numpyro")`); GPU path. |
| **nutpie** | `import nutpie` | A Rust/Numba NUTS sampler — often the fastest CPU option; great default for speed. |
| **BlackJAX** | `import blackjax` | A flexible JAX sampling library; research/experimentation backend. |
| **PyMC-BART** | `import pymc_bart as pmb` | Bayesian Additive Regression Trees (Chapter 14). |
| **CmdStanPy** | `from cmdstanpy import CmdStanModel` | The modern interface to Stan; we show Stan for a few canonical models. |

PyMC compiles your model to fast code via **PyTensor**, which needs a working C/C++ toolchain (and ideally a good BLAS). This is the single most common source of install pain, and it is exactly why we recommend conda-forge: it ships the compilers and an optimized math library so you don't fight your machine.

> ⚠️ **Pitfall: `pip install pymc` may pull PyMC 6.** As of mid-2026, PyMC 6.0.x exists and the official install page now defaults to `"pymc>=6"`. **This course is frozen to PyMC v5** — every idiom, every API spelling here is verified against v5 (the last v5 series is 5.28.x). If you let pip grab v6, some code may shift under you. So we **pin explicitly to v5** in every install line below. After installing, check `pm.__version__` and confirm it starts with `5.`, not `6.`.

### 3.2 Recommended: a fresh conda/mamba environment on conda-forge

Use a **fresh, dedicated environment**. Never install a research stack into your base environment — when something breaks (and something always breaks), you want to delete one environment, not your whole Python. `mamba` is a drop-in, much faster replacement for `conda`; if you have it, substitute `mamba` for `conda` below.

```bash
# Create a clean environment pinned to PyMC v5, from the conda-forge channel
conda create -c conda-forge -n bayes "pymc>=5.16,<6" arviz bambi numpyro nutpie cmdstanpy
conda activate bayes

# BART extension (pip is fine for this one)
pip install pymc-bart

# BlackJAX (optional JAX sampler)
pip install blackjax

# One-time: fetch the CmdStan toolchain so CmdStanPy can compile Stan models
python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"
```

That `install_cmdstan()` call downloads and builds the CmdStan toolchain (a few minutes, one time). You only need it for the Stan chapters/sections; if you skip Stan you can skip this step.

> 🔧 **In practice:** conda-forge is strongly preferred because it bundles the compiler toolchain plus MKL/BLAS, which sidesteps the most common PyTensor install failures. On macOS, if you go the pip route instead, install the Xcode Command Line Tools first (`xcode-select --install`). On Linux you need `g++`; on Windows the conda route is by far the least painful.

### 3.3 pip-only alternative

If you prefer pip (and you have a working C/C++ compiler for PyTensor):

```bash
pip install "pymc>=5.16,<6" arviz bambi "numpyro[cpu]" nutpie cmdstanpy pymc-bart blackjax
python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"
```

Note `"numpyro[cpu]"` — the `[cpu]` extra pulls a CPU build of JAX. The JAX/NumPyro/BlackJAX/nutpie backends are **optional**: you can complete the vast majority of this course on PyMC's default sampler alone, and only reach for the alternative backends in the scaling chapter (Chapter 17). If a backend import fails, it's almost always because the optional dependency isn't installed — install it separately and move on.

### 3.4 GPU and JAX (optional)

You do **not** need a GPU for this course. If you have one and want it, only the JAX backends (NumPyro, BlackJAX, and nutpie's JAX mode) use it. Follow JAX's own install instructions for your CUDA version (e.g. `pip install "jax[cuda12]"`). To run multiple chains in parallel on a single GPU, set `chain_method="vectorized"` — but note it is a backend-specific kwarg that must be passed *through* `nuts_sampler_kwargs`, not as a top-level `pm.sample` argument: `pm.sample(nuts_sampler="numpyro", nuts_sampler_kwargs={"chain_method": "vectorized"})`. (Chapter 17 covers the backend kwargs in full.) For most models in this course a GPU is overkill — modern CPU NUTS via nutpie or PyMC is plenty.

### 3.5 The 5-line "hello Bayes" smoke test

Here is the moment of truth. This tiny program builds a one-parameter model, samples it, and prints a summary. If it runs and prints a table, your environment works.

```python
import pymc as pm
import arviz as az

RANDOM_SEED = 8927  # we set a seed everywhere in this course for reproducibility

with pm.Model() as model:                       # 1. open a model context
    mu = pm.Normal("mu", mu=0, sigma=1)          # 2. a prior on a single unknown
    pm.Normal("y", mu=mu, sigma=1, observed=[0.3, -0.1, 0.8, 0.2])  # 3. likelihood + data
    idata = pm.sample(1000, tune=1000, chains=4, # 4. fit with NUTS → InferenceData
                      target_accept=0.9, random_seed=RANDOM_SEED)

print(az.summary(idata))                          # 5. read the posterior
```

When you run this you'll see PyMC report that it auto-assigned the **NUTS** sampler, then a progress bar across 4 chains, and finally a table from `az.summary` with columns including `mean`, `sd`, `hdi_3%`, `hdi_97%`, `ess_bulk`, `ess_tail`, and `r_hat`. For this trivial model `mu` should come out near **0.24** — slightly pulled toward the prior mean of 0 from the data mean of 0.3, because with only 4 data points the `Normal(0, 1)` prior still has a say. (Exactly: the conjugate posterior mean is $\frac{n\bar y}{n + \sigma^2_{\text{lik}}/\sigma^2_{\text{prior}}} = \frac{4 \cdot 0.3}{4 + 1} = 0.24$, with posterior SD $\approx 1/\sqrt{5} \approx 0.45$.) That little bit of shrinkage is your very first glimpse of Bayesian regularization — don't expect the raw data mean. And `r_hat` should read `1.00`. If you see that table, congratulations — you have a working Bayesian computing environment, and you've already done a full define → fit → summarize cycle.

> 🩺 **Diagnostic:** glance at the `r_hat` column right now and burn this into memory: you want it at **1.00** (we accept up to **1.01**). Anything higher means your chains disagree with each other — a red flag we'll dissect in Chapter 07. Even in a smoke test, the habit of looking at `r_hat` before anything else is one worth starting today.

> ⚠️ **Pitfall: v3/v4 idioms that no longer exist.** If you've seen older PyMC tutorials, unlearn these now. `pm.sample` returns an **`InferenceData`** object — we call it `idata`, never "the trace" (that was the v3 name; we'll note it once and never again). There is no `pm.sample_ppc` (it's `pm.sample_posterior_predictive`), no `pm.MutableData`/`pm.ConstantData` split (just `pm.Data`, mutable by default), and `pm.sample_prior_predictive` takes `draws=`, not `samples=`. We use only the modern v5 spellings.

---

## 4. A minute of theory you'll need everywhere

Before the workflow, the one equation that underlies all of it. Bayesian inference is the act of turning a prior belief into a posterior belief by conditioning on data. For data $y$ and parameters $\theta$:

$$ p(\theta \mid y) = \frac{p(y \mid \theta)\,p(\theta)}{p(y)}, \qquad p(y) = \int p(y\mid\theta)\,p(\theta)\,d\theta. $$

In words, and in the ASCII form we'll use for skimming:

```
posterior  ∝  likelihood × prior
p(θ|y)     ∝  p(y|θ)     × p(θ)
```

Each piece has a name you'll use constantly: $p(\theta)$ is the **prior** (what you believe before data), $p(y\mid\theta)$ is the **likelihood** (how probable the data are for given parameters — a function *of* $\theta$, not a distribution *over* $\theta$), $p(\theta\mid y)$ is the **posterior** (your updated belief), and $p(y)$ is the **evidence** or **marginal likelihood** (a normalizing constant, and the hard integral that makes us reach for MCMC). The $\propto$ form is what we actually compute, because the evidence cancels.

Two more distributions appear throughout the workflow, and naming them now will make §5 click:

$$ \underbrace{p(\tilde y) = \int p(\tilde y\mid\theta)\,p(\theta)\,d\theta}_{\text{prior predictive}}, \qquad \underbrace{p(\tilde y \mid y) = \int p(\tilde y\mid\theta)\,p(\theta\mid y)\,d\theta}_{\text{posterior predictive}}. $$

The **prior predictive** is fake data simulated from your model *before* seeing real data — sample $\theta$ from the prior, then $\tilde y$ from the likelihood. The **posterior predictive** is fake data simulated *after* fitting — sample $\theta$ from the posterior, then $\tilde y$. These two are not academic curiosities; they are literally the "check" steps of the workflow. We sample from the first to ask "are my priors sane?" and from the second to ask "does my fitted model reproduce reality?" Chapter 01 derives all of this carefully; here you just need the names.

> 💡 **Intuition:** a Bayesian model is a **joint distribution** $p(y, \theta) = p(y\mid\theta)\,p(\theta)$ — a recipe you can run *forward* (sample parameters, then data → prior predictive) or *backward* (condition on observed data → posterior). Everything else is bookkeeping on top of this one idea. Hold onto it.

---

## 5. The Bayesian workflow at a glance

Here is the loop that organizes the entire course. Read it top to bottom, then read it again as a cycle — because the defining feature of real Bayesian work is that you go *around* this loop several times, not through it once.

```
        ┌───────────────────────────────────────────────────────────────┐
        │                                                               │
        ▼                                                               │
   (1) DOMAIN KNOWLEDGE  ─────►  (2) GENERATIVE MODEL                   │
   what do you know about            write the joint p(y,θ):            │
   the problem, the scales,          priors p(θ) + likelihood p(y|θ)    │
   the plausible effects?                  │                           │
                                            ▼                           │
                              (3) PRIOR PREDICTIVE CHECK                │
                              simulate ỹ ~ p(ỹ) BEFORE seeing data.     │
                              Are the fake datasets physically sane?    │
                              (heights of 9 km ⇒ go back to step 2)     │
                                            │                           │
                                            ▼                           │
                                       (4) FIT                          │
                              run MCMC/NUTS (or VI / quadratic);        │
                              get the posterior p(θ|y) as InferenceData │
                                            │                           │
                                            ▼                           │
                              (5) DIAGNOSTICS  ("did the machine work?")│
                              R̂ ≤ 1.01,  ESS ≳ 400,  0 divergences,     │
                              healthy energy/BFMI.  If bad ──► step 2/7 │
                                            │   (the "folk theorem":    │
                                            │    bad sampling usually   │
                                            │    means a bad model)     │
                                            ▼                           │
                       (6) POSTERIOR PREDICTIVE CHECK                   │
                       simulate ỹ ~ p(ỹ|y); does the fitted model       │
                       reproduce features of the REAL data?            │
                                            │                           │
                                            ▼                           │
                       (7) EXPAND / COMPARE  ──────────────────────────►┘
                       add structure, compare models (LOO/WAIC),
                       loop back. The model is a draft, not a verdict.
                                            │
                                            ▼
                       (8) SBC  (simulation-based calibration)
                       the rigorous check that your inference machine
                       is itself calibrated, across many simulated
                       datasets — done once you trust a model class.
```

Let me walk each step, because the *why* of each one is what separates a checklist-follower from someone who actually thinks.

**(1) Domain knowledge.** Before any code, ask what you know. What are the units? What's a plausible magnitude for an effect? Is the outcome a count, a proportion, a continuous measurement? This is where priors and likelihood families are born. Skipping this is how you end up with a vague prior that allows the absurd.

**(2) Generative model.** Translate that knowledge into a joint distribution: priors on parameters plus a likelihood for the data. The acid test is McElreath's: *could you run this model forward to simulate fake data?* If yes, you have a generative model. If you can't say how data would come out, you don't yet have a model — you have a wish.

**(3) Prior predictive check.** Now *do* run it forward. Sample $\theta$ from the prior, push it through the likelihood, and look at the fake datasets. This catches the classic disaster — a vague prior on an unstandardized predictor implying human heights in the millions of centimeters. In PyMC: `pm.sample_prior_predictive(draws=500, random_seed=RANDOM_SEED)`, then plot with `az.plot_ppc(idata, group="prior")`. If the implied data are physically impossible, you go back to step 2. You have not earned the right to look at real data yet.

**(4) Fit.** Condition on the data to get the posterior. For nearly everything in this course that means MCMC with the NUTS sampler — `pm.sample(1000, tune=1000, chains=4, target_accept=0.9, random_seed=RANDOM_SEED)` — which returns an `InferenceData` object. (Chapters 04–06 cover the alternatives: conjugate/grid math, quadratic approximation, and variational inference, and exactly when each is the right tool.)

**(5) Diagnostics.** Ask: *did the machine actually work?* This is non-negotiable and it comes **before** you interpret a single posterior number. You check $\hat{R}$ (rank-normalized; want $\le 1.01$), effective sample size (`ess_bulk` and `ess_tail`, want $\gtrsim 400$), the number of **divergences** (want exactly 0), and the energy/BFMI diagnostics. Chapter 07 defines each of these precisely and shows the plots — trace, rank, forest, energy. The deep lesson, Gelman's **"folk theorem of statistical computing,"** is that *computational trouble usually signals a model problem*. A sampler that diverges is not being difficult; it is telling you the truth about a pathological geometry, and the fix is usually to change the model (e.g. a non-centered reparameterization), not to crank a knob.

**(6) Posterior predictive check.** Now ask: *does the fitted model reproduce reality?* Simulate $\tilde y$ from the posterior predictive and overlay it on the observed data — `pm.sample_posterior_predictive(idata, extend_inferencedata=True)` then `az.plot_ppc(idata)`. If your model says the data should be symmetric but the real data are skewed, you'll see it here. Betancourt calls a passing check **retrodictive** — the model *retrodicts* the data already in hand — and we'll use his term because it correctly stresses that this is a check against known data, not a forecast.

**(7) Expand / compare.** A model is a draft, not a verdict. If checks reveal a gap, add structure — a heavier-tailed likelihood, a hierarchical level, an interaction — and loop back. When you have several candidate models, compare them honestly with cross-validation (PSIS-LOO and WAIC, `az.loo` and `az.compare`; Chapter 11). Gelman's framing is that you traverse a whole *topology* of models, not a single one.

**(8) Simulation-based calibration (SBC).** The rigorous capstone check, due to Talts et al. (2018). Once you trust a *model class*, you validate the *inference algorithm itself*: draw $\theta^{(i)}$ from the prior, simulate data $y^{(i)}$, refit, and compute the **rank** of $\theta^{(i)}$ within its posterior draws. If inference is correct, those ranks are **uniform**; characteristic non-uniform shapes (∪, ∩, skew) diagnose specific failures. This is the gold standard, and Chapter 08 shows it in practice.

> 📜 **Citation/Origin:** this loop reconciles two canonical sources. **Gelman et al., "Bayesian Workflow" (2020, arXiv:2011.01808)** is the *descriptive, iterative* view — a rich toolbox and the folk theorem. **Betancourt's "Towards A Principled Bayesian Workflow"** is the *prescriptive, sequential* view — four evaluating questions (Domain Expertise Consistency, Computational Faithfulness, Inferential Adequacy, Model Adequacy) and fifteen disciplined steps that calibrate the algorithm *before* you ever touch real data. They agree on the spine above; Chapter 08 lays them side by side. For now: memorize the spine, and let the two authors be two zoom levels on it.

> 🔧 **In practice:** "**fit fast, fail fast.**" Don't polish a model in your head for an hour. Build the simplest version, run a short chain, and let the diagnostics tell you what's wrong — early failures are cheap and informative. Start simple, start complex, and meet in the middle.

---

## 6. The master dataset index

Examples *rhyme* across this course on purpose. When the same dataset reappears in a new chapter, you bring intuition with you and we don't waste your attention re-explaining the columns. Below is the shared catalog, each with a one-line load you can run today. The first time a dataset appears in a chapter we give the full load; later chapters assume familiarity and cross-reference here.

| Dataset | One-line load | Best for (chapters) |
|---|---|---|
| **Howell !Kung** (height/weight) | `pd.read_csv("https://raw.githubusercontent.com/rmcelreath/rethinking/master/data/Howell1.csv", sep=";")` | linear regression, first models, priors on slope/intercept (Ch 01–03, 09) |
| **Eight Schools** | `y=np.array([28,8,-3,7,-1,1,18,12]); sigma=np.array([15,10,16,11,9,11,10,18])` | the canonical hierarchical model, funnel, non-centering (Ch 07, 10) |
| **Radon (Minnesota)** | `pd.read_csv(pm.get_data("srrs2.dat"))` + `pd.read_csv(pm.get_data("cty.dat"))`, filter to `"MN"` | multilevel / partial pooling flagship (Ch 10) |
| **Palmer Penguins** | `sns.load_dataset("penguins")` | logistic/GLM, robust regression, missing data (Ch 09, 16) |
| **Titanic** | `sns.load_dataset("titanic")` | logistic regression + genuine missing data in `age` (Ch 09, 16) |
| **Bike sharing** | Bambi example data, `bikes` *(loaded in Ch 09; exact loader given there — verify in current Bambi docs)* | Poisson & Negative-Binomial count GLM, splines (Ch 09, 14) |
| **Chimpanzees** (prosocial) | `pd.read_csv(base + "chimpanzees.csv", sep=";")` | hierarchical logistic, varying intercepts (Ch 10) |
| **Baseball batting** (Efron–Morris) | small hardcoded table (or `pybaseball`) | shrinkage / partial pooling intuition (Ch 10) |
| **Mauna Loa CO₂** | `sm.datasets.co2.load_pandas().data` (statsmodels) | Gaussian processes, structural time series (Ch 12, 15) |
| **Airline passengers** | `sns.load_dataset("flights")` (this *is* AirPassengers, long format) | time series, trend + seasonality (Ch 15) |
| **Coal mining disasters** | `pd.read_csv(pm.get_data("coal.csv"), header=None)` (two NaN years!) | change-point / Poisson, auto-imputation bridge (Ch 13, 15, 16) |
| **Old Faithful / geyser** | `sns.load_dataset("geyser")` | mixture models, density estimation (Ch 13) |
| **Tips / Diamonds** | `sns.load_dataset("tips")` / `sns.load_dataset("diamonds")` | GLMs, interactions, robust & log models (Ch 09) |
| **Rugged / GDP** (terrain) | `pd.read_csv(base + "rugged.csv", sep=";")` | interactions, log models (Ch 09) |

A few practical notes, verified from the dataset dossier:

- The **`base`** in two rows above is the rethinking raw-CSV root: `base = "https://raw.githubusercontent.com/rmcelreath/rethinking/master/data/"`. These CSVs are **semicolon-separated** — a classic footgun if you forget `sep=";"`.
- **Seaborn** ships a generous set of built-ins. Confirmed-available names include `titanic`, `penguins`, `tips`, `geyser`, `diamonds`, `flights`, `fmri`, `mpg`, `iris`, and `planets`. The `flights` dataset is the famous monthly airline-passenger series (1949–1960) in long format — pivot `year`/`month`/`passengers` for a time index. Note on the missing-data examples: `penguins` carries only a handful of NaN rows (a good *light* missing-data demo), whereas Titanic's `age` column is the heavier, more realistic missing-data case — we use both, for different reasons.
- **PyMC built-ins** load via `pm.get_data("...")`: `coal.csv` (coal disasters; note it has **two NaN years**, which is a deliberate bridge to automatic missing-data imputation in Chapter 16), and `srrs2.dat`/`cty.dat` for radon. Verify exact filenames against your installed PyMC if a load ever errors.
- **ArviZ** ships ready-made `InferenceData` for the eight-schools model — `az.load_arviz_data("centered_eight")` and `"non_centered_eight"` — which is wonderful for *demonstrating diagnostics without sampling*. We use it in Chapter 07.

> 🔧 **In practice: when a URL is uncertain, generate synthetic data with a known truth.** This is not a fallback — it's a *superpower* for learning. When you control the generative process, you can check whether inference recovers the parameters you put in. Keep this snippet handy; you'll use it constantly:
> ```python
> import numpy as np
> rng = np.random.default_rng(8927)
> a_true, b_true, s_true = 2.0, 0.5, 1.0       # the truth WE chose
> x = rng.normal(size=100)
> y = rng.normal(a_true + b_true * x, s_true)  # data from a process we fully know
> # ... fit, then confirm the posterior covers a_true, b_true, s_true
> ```

---

## 7. Notation & conventions — a quick reference

Skim this now; refer back as needed. We keep notation rigidly consistent across all eighteen chapters so symbols never shift meaning under you.

**Core symbols.**

| Symbol | Meaning |
|---|---|
| $y$ | observed data |
| $\theta$ | parameters (the unknowns we infer) |
| $p(\theta)$ | prior |
| $p(y\mid\theta)$ | likelihood (a function of $\theta$, not a distribution over it) |
| $p(\theta\mid y)$ | posterior |
| $p(y)$ | marginal likelihood / evidence |
| $\tilde y$ | predictive data (prior or posterior predictive) |

**Distributions are written** $\theta \sim \mathrm{Normal}(\mu, \sigma)$.

> ⚠️ **Pitfall — the precision footgun.** PyMC and Stan parameterize the Normal by its **standard deviation $\sigma$**, *not* its variance and *not* its precision. So `pm.Normal("mu", mu=0, sigma=1)` is a unit-SD normal. If you've used **BUGS or JAGS**, beware: they parameterize by **precision** $\tau = 1/\sigma^2$, so a "Normal(0, 1)" there means precision 1, i.e. SD 1 — but "Normal(0, 0.0001)" there means a *very wide* prior, not a tight one. This single difference has burned generations of practitioners. In this course, the second argument to a Normal is always a standard deviation.

**Standard prior choices** (developed fully in Chapters 02–03, previewed here so the code reads naturally):

- **Scale/variance parameters** (anything that must be positive, like $\sigma$): `pm.HalfNormal`, `pm.HalfCauchy`, or `pm.Exponential`. Never put a plain Normal on a standard deviation.
- **Correlation matrices** (for correlated varying effects): the **LKJ** prior, `pm.LKJCholeskyCov`.
- **Location parameters** (intercepts, slopes): typically `pm.Normal`, on a scale set by *standardized* predictors.

> 💡 **Standardize your predictors by default.** Throughout this course we standardize continuous predictors (subtract the mean, divide by the SD) before modeling. The reason is that it puts every coefficient on a common, interpretable scale, which makes choosing sane weakly-informative priors *easy* — a `Normal(0, 1)` on a standardized slope is a reasonable default rather than a wild guess. Chapter 03 establishes this; every later chapter assumes it.

**Code conventions you'll see in every chapter.**

- `RANDOM_SEED = 8927` set at the top of each example and passed to `pm.sample(..., random_seed=RANDOM_SEED)` and the predictive functions, for reproducibility.
- Models built with **named dims/coords**: `with pm.Model(coords=coords) as model:` and `dims="obs_id"` on variables, so `az.summary` and plots come out labeled and readable.
- `pm.Data("x", values, dims=...)` for swap-in prediction (mutable by default in v5; update with `pm.set_data({...})`).
- `pm.Deterministic("name", expr)` to record derived quantities (like `mu = a + b*x`) in the InferenceData.
- The default sampler call is `pm.sample(1000, tune=1000, chains=4, target_accept=0.9, random_seed=RANDOM_SEED)`; we raise `target_accept` to 0.95–0.99 when wrestling divergences.
- Inference results are always **`InferenceData`** (we call it `idata`), never "the trace."
- Anything not meant to run is explicitly marked `# pseudocode`.

**Diagnostic thresholds** (defined precisely in Chapter 07; quoted consistently everywhere):

| Diagnostic | Target | Meaning |
|---|---|---|
| $\hat{R}$ (rank-normalized) | $\le 1.01$ | chains agree; $> 1.01$ means non-convergence |
| ESS (bulk & tail) | $\gtrsim 400$ (≳100/chain) | enough effective draws for stable estimates |
| Divergences | $0$ | any divergence signals a geometry/model problem |
| $\hat{k}$ (PSIS-LOO) | $< 0.7$ | LOO estimate trustworthy; $\ge 0.7$ unreliable |
| BFMI | $\ge 0.3$ | energy diagnostic; $< 0.3$ flags trouble |

---

## 8. The recommended path through the course

Here is the map of the territory. The chapters are designed to be read in order — each builds on the last — but I'll also suggest shortcuts for readers in a hurry.

| Ch | Title | One-line what & why |
|---|---|---|
| **00** | Welcome & Setup | *(you are here)* the front door: philosophy, environment, workflow, datasets, notation. |
| **01** | Foundations of Bayesian Inference | Bayes' theorem decomposed; prior/likelihood/posterior/evidence; generative modeling; small-world vs large-world; Bayesian vs frequentist, honestly. |
| **02** | Priors 1 — Concepts & Common Priors | What a prior *is*; flat vs proper; conjugate, weakly-informative, regularizing, PC priors; the catalog of distributions and when to reach for each. |
| **03** | Priors 2 — Choosing, Building & Checking | Prior predictive checks; standardization & scaling; priors for scale/variance params; eliciting priors; sensitivity analysis; "should I look at the data first?" |
| **04** | Inference 1 — Analytic, Grid & Quadratic | Conjugate/analytic posteriors; grid approximation; Laplace/quadratic approximation (QUAP); when each works and fails. |
| **05** | Inference 2 — MCMC from Metropolis to NUTS | Markov chains; Metropolis–Hastings; Gibbs; the typical set & curse of dimensionality; Hamiltonian Monte Carlo; NUTS; warmup, step size, mass matrix, `target_accept`. |
| **06** | Inference 3 — Variational Inference | KL and the ELBO; mean-field ADVI; full-rank; normalizing-flow VI; Pathfinder; VI vs MCMC; where VI lies to you and how to catch it. |
| **07** | MCMC Diagnostics & Debugging | $\hat{R}$, ESS, MCSE; trace/rank/forest plots; divergences; energy/BFMI; the funnel; **centered vs non-centered** reparameterization; the ArviZ diagnostic loop. |
| **08** | The Bayesian Workflow | Gelman et al. + Betancourt reconciled; prior predictive → fit → diagnose → posterior predictive → expand/compare → **SBC** and fake-data recovery. |
| **09** | GLMs & Bambi | Linear, logistic, Poisson/Negative-Binomial, robust (Student-t), ordinal; link functions & interpretation; the **Bambi** formula API with raw-PyMC equivalents; contrasts. |
| **10** | Hierarchical & Multilevel Models | Partial pooling; varying intercepts & slopes; the **radon** flagship; shrinkage; hyperpriors; non-centered parameterization in depth; LKJ correlations. |
| **11** | Model Comparison & Evaluation | The predictive view; ELPD; **LOO-CV via PSIS**; WAIC; the $\hat{k}$ diagnostic; `az.compare`; stacking & model averaging; the trouble with Bayes factors. |
| **12** | Gaussian Processes | Kernels; GP regression (`pm.gp.Marginal`); latent GPs (`pm.gp.Latent`); priors on lengthscale/amplitude; combining kernels; sparse & **HSGP** approximations. |
| **13** | Mixture Models & Latent Variables | Finite mixtures; identifiability & label switching (ordered constraint); marginalizing discrete latents; zero-inflated & hurdle models. |
| **14** | Nonparametric & Flexible Models | Bayesian splines/basis functions; **BART** (`pymc_bart`); Dirichlet-process/infinite mixtures (stick-breaking); GPs vs BART vs splines. |
| **15** | Time Series & State-Space Models | AR; Gaussian random walk; structural TS (trend + seasonality); stochastic volatility; the Kalman/state-space view; change-points; forecasting with uncertainty. |
| **16** | Missing Data, the Bayesian Way | MCAR/MAR/MNAR; imputation as parameters; PyMC automatic imputation; modeling the missingness mechanism; censoring & truncation as relatives. |
| **17** | Scaling & Advanced Computation | Backends (NumPyro/JAX, nutpie, BlackJAX); GPU; within-chain parallelism (`reduce_sum`); minibatch ADVI; marginalization for speed; picking a sampler. |
| **18** | Capstone Case Studies | Two or three end-to-end studies chaining the *whole* workflow on real data — every step, start to finish. |
| **A** | Appendix — Cheatsheets & Resources | Distribution cheat-sheet; prior cheat-sheet; diagnostic thresholds; troubleshooting table; glossary; master resource list; what-to-learn-next roadmap. |

**Suggested routes:**

- **The full climb (recommended).** 00 → 18 in order, running every example. This is the way to become genuinely expert.
- **"I need to ship a GLM next week."** 00 → 01 → 02 → 03 → 05 → 07 → 09, then 11 for comparison. You'll have priors, sampling, diagnostics, and GLMs.
- **"I came for hierarchical models."** Do the GLM route, then 10 (the heart of the matter) and 07's non-centering section, which 10 leans on heavily.
- **"I came for the compute."** 05 → 06 → 07 give you the inference engines; 17 covers scaling and backends.
- **Reference mode.** Keep the Appendix (A) open in a tab. Its cheat-sheets and troubleshooting table are the things you'll reach for at 2am.

> 🔧 **In practice:** whatever route you take, do not skip **Chapter 07 (Diagnostics)**. It is the chapter that makes every other chapter trustworthy. A model you can't diagnose is a model you can't believe.

---

## 9. A tiny end-to-end taste (the whole workflow in about thirty lines)

You'll see fully worked examples on named datasets throughout the course. But let me give you a complete, runnable taste *right now* — every step of the workflow on synthetic data with a **known truth**, so you can watch inference recover the parameters we put in. This is the simplest possible "owl," drawn in one go; later chapters draw it slowly, step by step.

We'll simulate a simple linear regression with known coefficients, then fit it and confirm the posterior covers the truth. One thing to notice up front: we draw `x` from `rng.normal(size=100)`, so the predictor is already on a unit scale — effectively standardized. That is exactly why a `Normal(0, 1)` slope prior is sane here without any extra step. With raw, wide-ranging predictors (say, income from 0 to 1,000,000) you would standardize *first* — subtract the mean, divide by the SD — before putting that same prior on the slope (Chapter 03 makes this a default).

```python
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)

# --- (1)-(2) Generative model: simulate data from a process WE control ---
a_true, b_true, s_true = 2.0, 0.5, 1.0
x = rng.normal(size=100)                       # x is already unit-scale (effectively standardized)
y = rng.normal(a_true + b_true * x, s_true)    # y ~ Normal(a + b*x, sigma)

coords = {"obs_id": np.arange(len(y))}
with pm.Model(coords=coords) as model:
    x_data = pm.Data("x", x, dims="obs_id")            # mutable; swap for prediction later
    a = pm.Normal("a", mu=0, sigma=3)                  # priors are sane because x is unit-scale
    b = pm.Normal("b", mu=0, sigma=1)
    sigma = pm.HalfNormal("sigma", sigma=2)            # scale param: positive-only prior
    mu = pm.Deterministic("mu", a + b * x_data, dims="obs_id")
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y, dims="obs_id")

    # --- (3) Prior predictive check (BEFORE fitting) ---
    idata = pm.sample_prior_predictive(draws=500, random_seed=RANDOM_SEED)

    # --- (4) Fit ---
    idata.extend(pm.sample(1000, tune=1000, chains=4, target_accept=0.9,
                           random_seed=RANDOM_SEED))

    # --- (6) Posterior predictive check ---
    pm.sample_posterior_predictive(idata, extend_inferencedata=True,
                                   random_seed=RANDOM_SEED)

# --- (5) Diagnostics + read the posterior ---
print(az.summary(idata, var_names=["a", "b", "sigma"]))

# --- The two visual checks (discussed below) ---
az.plot_ppc(idata, group="prior")   # prior predictive: are the priors sane BEFORE fitting?
az.plot_ppc(idata)                  # posterior predictive: does the fit reproduce the data?
plt.show()
```

**What you should see, and how to read it.** The `az.summary` table will have one row each for `a`, `b`, and `sigma`. The `mean` column should land near our truth — `a` ≈ 2.0, `b` ≈ 0.5, `sigma` ≈ 1.0 — and the `hdi_3%`/`hdi_97%` columns give a 94% credible interval that should *contain* those true values. That containment is the win: it means inference recovered the process we built. Crucially, check the last three columns first: `ess_bulk` and `ess_tail` should be comfortably in the thousands, and `r_hat` should read `1.00`. If `r_hat` were above 1.01 or you saw a divergence warning, you'd stop and debug *before* trusting a single mean.

Two plots make the checks visual (run them yourself):

- `az.plot_ppc(idata, group="prior")` — the **prior predictive** overlay. Before fitting, this shows the range of datasets your priors imply. With our weakly-informative priors the fake data should sit in a plausible range (roughly single digits), not span millions. If it spanned millions, that's your cue to tighten priors — back to step 2.
- `az.plot_ppc(idata)` — the **posterior predictive** overlay. After fitting, this shows simulated datasets from the posterior (thin lines) against the observed data (thick line). For a well-fitting model the observed density should sit comfortably inside the cloud of replications. A systematic mismatch — say, the real data clearly more skewed than every replication — is the signal to expand the model (a heavier-tailed likelihood, an extra predictor) and loop back to step 7.

> 🩺 **Diagnostic:** notice we computed the **prior** predictive *before* `pm.sample` and the **posterior** predictive *after*. That ordering is the workflow itself. The prior predictive earns you the right to fit; the diagnostics earn you the right to interpret; the posterior predictive earns you the right to believe. Skip a step and you've skipped your own permission to trust the result.

We deliberately ran this loop on *synthetic data with a known truth* — the cleanest way to confirm that inference recovers what we put in. Chapter 01 runs this very same loop on a **named** real dataset, the Howell !Kung heights, where there is no known truth to check against and every workflow step has to earn its keep.

That's the entire loop in one screen. Every chapter that follows is, in a sense, an elaboration of these few dozen lines — richer models, harder geometry, deeper diagnostics — but always this same define → fit → check spine.

---

## ⚠️ Common errors & how to fix them

These are the setup- and orientation-level traps. Modeling-specific errors live in their own chapters' tables; this one gets you to a working, correctly-pinned environment running modern v5 code.

| Symptom | Cause | Fix |
|---|---|---|
| `pip install pymc` pulls **PyMC 6.x** | The official install page now defaults to `pymc>=6`; the course is frozen to v5. | Pin explicitly: `conda create -c conda-forge -n bayes "pymc>=5.16,<6" ...` or `pip install "pymc>=5.16,<6"`. Verify `pm.__version__` starts with `5.`. |
| Slow/broken C compile, BLAS warnings on install | PyMC compiles via PyTensor, which needs a C/C++ toolchain and a good BLAS. | Prefer **conda-forge** (ships compilers + MKL). On macOS install Xcode CLT (`xcode-select --install`); on Linux ensure `g++`. |
| `AttributeError: module 'pymc' has no attribute 'MutableData'` / `'sample_ppc'` | v3/v4 idioms removed in v5. | Use `pm.Data` (mutable by default), `pm.sample_posterior_predictive`, and treat results as `InferenceData` — not "trace." |
| `sample_prior_predictive(samples=...)` raises | The argument is `draws=`, not `samples=`. | Call `pm.sample_prior_predictive(draws=500, random_seed=RANDOM_SEED)`. |
| Prior predictive draws look insane (heights of $10^6$ cm) | Vague priors on **unstandardized** predictors → small-world nonsense. | Standardize predictors (Ch 03), use weakly-informative priors, re-run the prior predictive check. |
| `import numpyro` / `import blackjax` fails | Optional JAX backend not installed. | `pip install "numpyro[cpu]"` (or `blackjax`); these are optional — the course runs on PyMC's default sampler. |
| `nutpie` not found | Optional Rust-compiled wheel not installed. | `conda install -c conda-forge nutpie` or `pip install nutpie`. |
| `CmdStanModel` can't find the Stan toolchain | CmdStan not fetched. | `pip install cmdstanpy` then `python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"` (one-time). |
| "My 95% confidence interval = 95% chance $\theta$ is inside it" | Conflating a frequentist CI with a Bayesian credible interval. | Only the **credible** interval licenses $P(\theta \in [\ell,u]\mid y)=0.95$. We unpack the fixed-vs-random distinction in Ch 01. |
| Different results every run | No random seed. | Set `RANDOM_SEED` and pass it to `pm.sample(..., random_seed=RANDOM_SEED)` and the predictive calls. |

---

## 🧪 Exercises

Work these in a notebook in your fresh `bayes` environment. Hints are deliberately light — struggle a little; it's where the learning is.

1. **(Setup, mandatory.)** Create the `bayes` environment, run the 5-line "hello Bayes" smoke test from §3.5, and confirm `az.summary` prints a table with `r_hat` equal to `1.00`. Then print `pm.__version__` and verify it starts with `5.` (not `6.`). *Hint: if it starts with `6`, you let pip ignore the pin — recreate the env with the `"pymc>=5.16,<6"` constraint.*

2. **(The six-question habit.)** Take the tiny linear-regression model in §9 and, in your own words, answer all six study questions from §2.2 for it: generative story, priors (and why those scales), likelihood, how to fit, how to diagnose, and one way it could break. *Hint: for "how it breaks," think about what happens if you forget to standardize a predictor that ranges from 0 to 1,000,000.*

3. **(Break the prior predictive.)** Re-run the §9 model but change the slope prior to `pm.Normal("b", 0, 1000)` and the predictor to `x = rng.normal(0, 1000, size=100)` *without* standardizing. Plot `az.plot_ppc(idata, group="prior")`. Describe what the implied datasets look like and explain, in workflow terms, why this means you should not proceed to fitting. *Hint: this is the "heights of $10^6$" failure in miniature.*

4. **(Recover the truth.)** Modify the §9 simulation to use `a_true, b_true, s_true = -1.0, 2.5, 0.5`. Re-fit and check whether the 94% HDIs for `a`, `b`, and `sigma` contain these new true values. *Hint: they should. If one doesn't, run a few more times with different seeds — occasional misses are expected, since a 94% interval misses ~6% of the time.*

5. **(Read the workflow.)** Without looking back at §5, write the eight-step workflow spine from memory, in order, and next to each step write the one question it answers (e.g. step 5 → "did the machine actually work?"). Then check yourself against §5. *Hint: the steps pair up as define (1–3), fit (4), check (5–6), improve (7–8).*

6. **(Conceptual, stretch.)** In one paragraph each, explain to a frequentist colleague (a) why a Bayesian *credible* interval lets you say "95% probability $\theta$ is in here" while their *confidence* interval does not, and (b) one concrete situation where going Bayesian changes the answer in practice. *Hint: for (b), think small samples where the prior actually contributes, or a hierarchy where partial pooling beats both "pool everything" and "pool nothing." Ch 01 develops both.*

---

## 📚 Resources & further reading

The canonical library, with specific entry points. Every "further reading" box in this course draws from this core list plus chapter-specific items.

**Books.**
- **McElreath, *Statistical Rethinking*, 2nd ed. (2020)** — the source of the golem / small-world / generative framing; Ch 1 ("The Golem of Prague") and Ch 2 ("Small Worlds and Large Worlds"). Book page: https://xcelab.net/rm/ ; free lectures: https://github.com/rmcelreath/stat_rethinking_2023 .
- **Martin, Kumar, Lao, *Bayesian Modeling and Computation in Python* (CRC, 2021)** — the PyMC/ArviZ companion to this course, free online at https://bayesiancomputationbook.com . Ch 1 (inference) and Ch 9 (workflow/end-to-end) parallel this chapter directly.
- **Martin, *Bayesian Analysis with Python*, 3rd ed. (2024)** — clean, idiomatic PyMC + ArviZ examples.
- **Gelman, Carlin, Stern, Dunson, Vehtari, Rubin, *Bayesian Data Analysis* (BDA3, 2013)** — the reference. Free PDF: http://www.stat.columbia.edu/~gelman/book/ .
- **Gelman, Hill, Vehtari, *Regression and Other Stories* (2020)** — superb on the Bayesian-vs-frequentist interval discussion; free: https://avehtari.github.io/ROS-Examples/ .
- **Kruschke, *Doing Bayesian Data Analysis*, 2nd ed.** — the gentle "puppy book."

**The workflow papers (read these around Chapter 08).**
- **Gelman et al., "Bayesian Workflow" (2020)** — https://arxiv.org/abs/2011.01808 — the descriptive, iterative view and the "folk theorem of statistical computing."
- **Betancourt, "Towards A Principled Bayesian Workflow"** — https://betanalpha.github.io/assets/case_studies/principled_bayesian_workflow.html — the four questions, fifteen steps, three phases. His writing index (HMC, hierarchical, identifiability case studies): https://betanalpha.github.io/writing/ .
- **Talts, Betancourt, Simpson, Vehtari, Gelman, "Validating Bayesian Inference Algorithms with Simulation-Based Calibration" (2018)** — https://arxiv.org/abs/1804.06788 — the definition of SBC and the rank-is-uniform result.

**Docs & install.**
- **PyMC installation page** — https://www.pymc.io/projects/docs/en/latest/installation.html — conda/mamba + optional backends. (⚠️ It now defaults to `pymc>=6`; pin to v5 as we do.)
- **PyMC "Prior and Posterior Predictive Checks" core notebook** — https://www.pymc.io/projects/docs/en/latest/learn/core_notebooks/posterior_predictive.html — the verified v5 idioms (`draws=`, `extend_inferencedata=`, `predictions=`, group names).
- **PyMC example gallery** — https://www.pymc.io/projects/examples/en/latest/gallery.html ; the GLM linear-regression core notebook is a clean end-to-end template: https://www.pymc.io/projects/docs/en/latest/learn/core_notebooks/GLM_linear.html .
- **ArviZ docs** — https://python.arviz.org/ — `plot_ppc`, `plot_trace`, `summary`, the InferenceData schema.
- **Bambi docs** — https://bambinos.github.io/bambi/ — the formula API for the GLM chapter.

**Philosophical anchor.**
- **Jaynes, *Probability Theory: The Logic of Science* (2003)** — Ch 1–2 derive probability as the unique calculus of rational belief (Cox's theorem); the bedrock for "why Bayesian."

---

## ➡️ What's next

You have a working environment, the six-question study habit, the workflow spine, the dataset index, and the notation. That's the whole front door — everything from here builds on it.

In **Chapter 01 — Foundations of Bayesian Inference**, we slow down and earn every piece of the equation we glimpsed in §4. We'll treat probability as extended logic (Cox and Jaynes), decompose Bayes' theorem term by term, build our very first real model on the **Howell !Kung** height data, and confront the question that quietly divides the whole field: what is *actually* different between Bayesian and frequentist inference, and when does it change your answer? We'll meet McElreath's golem and the small-world/large-world distinction that justifies the entire workflow you just learned. Bring the six questions with you — Chapter 01 is where you'll answer them for real, for the first time.
