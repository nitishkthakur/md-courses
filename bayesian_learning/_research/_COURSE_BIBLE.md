# COURSE BIBLE — Bayesian Learning (internal authoring guide, not shipped to learner)

> Every writing subagent MUST read this file first, then read the research dossier(s)
> assigned to its chapter, then write its chapter to the specified file. This file
> guarantees coherence: shared voice, shared libraries/versions, shared datasets,
> shared notation, and an accurate cross-reference map so chapters can refer to each
> other by number/title without contradiction.

---

## 1. What this course is

A complete, expert-level, practice-first course on **Bayesian modeling and the Bayesian
workflow** in Python. The learner is a strong ML practitioner who wants to become a
genuine expert — theory *and* hands-on. The course must leave them able to build,
fit, diagnose, debug, compare, and ship Bayesian models across the full zoo (GLMs,
hierarchical, GPs, mixtures, nonparametric/BART, time series, missing data), choosing
priors deliberately and reading every diagnostic.

The course is delivered as a folder of numbered Markdown chapter files. It is the
"theory + judgment + runnable code" layer — like a great textbook you can execute.

## 2. The teacher persona & voice (NON-NEGOTIABLE)

Write as a **state-of-the-art Bayesian researcher and practitioner mentoring a talented
junior**. Blend four teaching styles deliberately:

- **Richard McElreath** (*Statistical Rethinking*): build deep intuition first, golems
  and small-worlds metaphors, generative thinking, "the model is a small world,"
  relentless plots, owls drawn step by step. Warm, vivid, story-driven.
- **Michael Betancourt** (betanalpha case studies): mathematical rigor, the *principled*
  workflow, geometry of the typical set, why HMC works, divergences as truth-tellers.
- **Osvaldo Martin** (*Bayesian Analysis with Python*; *Bayesian Modeling and Computation
  in Python*): clean, idiomatic PyMC + ArviZ; pragmatic; lots of small complete examples.
- **Ravin Kumar / Junpeng Lao / Thomas Wiecki** (ArviZ, PyMC core devs): diagnostics-first,
  InferenceData, real production judgment.

Concrete voice rules:
- Open every chapter with a short **"A note from me to you"** block (italic blockquote)
  that frames *why this chapter matters* and what painful mistake it prevents.
- Talk *to* the learner ("you", "we", "let me show you", "sit with this for a second").
- Be **verbose and generous** — explain the *why* behind every choice, not just the how.
  When you introduce a concept, go all the way in. Do not hand-wave. Do not say "it can
  be shown that"; show it or give the honest intuition.
- Use callouts liberally (see §6).
- Every nontrivial claim that a skeptic would challenge gets a reason or a citation.
- Never be dry. This should read like the best lecture the learner has ever attended.

## 3. Library & version policy (write code that RUNS on these)

Primary stack (assume installed; pin mentally to these modern APIs):
- **PyMC v5** (`import pymc as pm`) — the primary modeling library for almost all examples.
- **ArviZ** (`import arviz as az`) — ALL diagnostics, summaries, and plots. Models return
  `az.InferenceData`. Use `az.summary`, `az.plot_trace`, `az.plot_rank`, `az.plot_ppc`,
  `az.plot_energy`, `az.loo`, `az.compare`, `az.plot_forest`, `az.plot_posterior`.
- **Bambi** (`import bambi as bmb`) — show for GLMs / multilevel formula-style models
  ("whenever you can use Bambi, sometimes show Bambi"). Always *also* show the equivalent
  raw-PyMC model so the learner sees what Bambi generates.
- **Stan via CmdStanPy** (`from cmdstanpy import CmdStanModel`) — show Stan for at least:
  one full model (GLM or hierarchical), marginalizing discrete parameters (mixtures /
  missing data), and `reduce_sum` within-chain parallelism. Mention PyStan exists but use
  CmdStanPy as the modern interface. Stan code in ```stan blocks.
- **NumPyro / JAX** (`import numpyro`) and **BlackJAX**, **nutpie** — for the scaling/GPU/
  speed chapter and where a JAX backend matters (`pm.sample(nuts_sampler="numpyro")` /
  `"nutpie"` / `"blackjax"`).
- **PyMC-BART** (`import pymc_bart as pmb`) for BART.
- Standard: `numpy as np`, `pandas as pd`, `matplotlib.pyplot as plt`, occasionally
  `scipy.stats as stats`.

PyMC v5 idioms to use consistently (do NOT use removed v3 APIs):
- `with pm.Model(coords=coords) as model:` ... use **named dims/coords** for readable output.
- Distributions: `pm.Normal("mu", mu=0, sigma=1)`, `pm.HalfNormal`, `pm.Exponential`,
  `pm.StudentT`, `pm.Gamma`, `pm.Beta`, `pm.Dirichlet`, `pm.LKJCholeskyCov`,
  `pm.MvNormal`, `pm.Binomial`, `pm.Poisson`, `pm.NegativeBinomial`, `pm.Bernoulli`.
- Data containers: `pm.Data("x", x_values)` (mutable by default in v5) for swap-in prediction.
- Deterministics: `pm.Deterministic("name", expr)`.
- Sampling: `idata = pm.sample(1000, tune=1000, target_accept=0.9, chains=4,
  random_seed=RANDOM_SEED)` → returns InferenceData. Prior: `pm.sample_prior_predictive`.
  Posterior predictive: `pm.sample_posterior_predictive(idata, extend_inferencedata=True)`.
- GP: `pm.gp.Latent`, `pm.gp.Marginal`, `pm.gp.HSGP`, covariance `pm.gp.cov.ExpQuad`, etc.
- Non-centered: write explicitly with `offset = pm.Normal(...); x = mu + offset*sigma` or
  use the `pm.Normal(..., transform=...)`/`ZeroSumNormal` where relevant — show the manual
  form for teaching.
- Always set `RANDOM_SEED = 8927` (or similar) at top and pass it for reproducibility.
- Mark anything not runnable as `# pseudocode` explicitly.

If unsure whether an exact API symbol exists in v5, prefer the well-established spelling
above and/or note "check the current PyMC docs" — never invent function names.

## 4. Required chapter structure (every chapter)

1. `# Chapter N — Title`
2. A one-line subtitle in `###` or bold under the title.
3. **"A note from me to you"** italic blockquote (the framing/why).
4. **What you'll be able to do after this chapter** — a short bulleted learning-objectives list.
5. The body: deep narrative sections with `##`/`###` headers. Interleave intuition → math →
   code → diagnostics → interpretation. Use real numbers where possible.
6. At least one **fully worked end-to-end example** on a named dataset from §7, including:
   data load/preprocessing → model → prior predictive check → sample → diagnostics →
   posterior predictive check → interpretation/post-processing (intervals, contrasts,
   predictions). Describe what the key plots show (the learner will run them).
7. **⚠️ Common errors & how to fix them** — a box/table mapping symptom → cause → fix,
   specific to the chapter's models/algorithms.
8. **🧪 Exercises** — 3–6 graded exercises (some conceptual, some coding), with hints.
9. **📚 Resources & further reading** — curated, with *specific* pointers (book + chapter,
   paper + author/year, blog/case-study + URL, library doc page). Use the canonical list
   in §8 plus chapter-specific items from the research dossier. Every chapter ends with this.
10. **➡️ What's next** — one-paragraph bridge to the next chapter by number and title.

Length: be generous. Target **3,500–9,000+ words** per content chapter (more for the big
ones: priors, MCMC/NUTS, diagnostics, hierarchical, GPs). Build the file incrementally:
`Write` the first portion, then `Edit`/append subsequent sections so length is not capped
by a single message. Do not pad with fluff — depth, not filler.

## 5. Notation conventions

- Use `$...$`/`$$...$$` LaTeX for math (GitHub renders it). Also give ASCII forms for key
  equations when it aids skimming, mirroring the recsys course.
- $y$ data, $\theta$ parameters, $p(\theta)$ prior, $p(y\mid\theta)$ likelihood,
  $p(\theta\mid y)$ posterior, $p(y)$ marginal/evidence, $\tilde y$ predictive.
- Distributions written $\theta \sim \mathrm{Normal}(\mu,\sigma)$ — **PyMC/Stan use
  standard deviation $\sigma$, not variance, and not precision**; say so explicitly and
  warn that BUGS/JAGS used precision (a classic footgun).
- Half-Normal/half-Cauchy/Exponential for scale params; LKJ for correlation matrices.

## 6. Callout convention (use these emoji headers inline)

- `> 💡 **Intuition:**` — the mental model.
- `> 🔧 **In practice:**` — practical/engineering guidance, knobs.
- `> ⚠️ **Pitfall:**` — the trap juniors fall into.
- `> 🩺 **Diagnostic:**` — how to read a specific diagnostic/plot.
- `> 🧮 **The math:**` — a derivation or precise statement.
- `> 📜 **Citation/Origin:**` — where an idea/paper comes from.

## 7. Shared dataset catalog (pull examples from these for consistency)

Use these so examples rhyme across chapters. Prefer ones with easy loads. Give the load
snippet the first time a dataset appears; later chapters can assume familiarity and cross-ref.

| Dataset | Load | Best for |
|---|---|---|
| **Howell !Kung (height/weight)** | `pd.read_csv(".../Howell1.csv", sep=";")` (rethinking repo) | linear regression, first models, priors on slope/intercept (Ch 1–3, 9) |
| **Eight Schools** | hardcoded `y=[28,8,-3,7,-1,1,18,12]`, `sigma=[15,10,16,11,9,11,10,18]` | the canonical hierarchical model, funnel, non-centering (Ch 7, 10) |
| **Radon (Minnesota)** | from Gelman/Hill; PyMC example data / `pd.read_csv` | multilevel/partial pooling flagship (Ch 10) |
| **Palmer Penguins** | `import palmerpenguins` or seaborn `load_dataset` proxy | logistic/GLM, robust regression, missing data (Ch 9, 16) |
| **Titanic** | seaborn `sns.load_dataset("titanic")` | logistic regression + genuine missing data (Ch 9, 16) |
| **Bike sharing / `bikes`** | Bambi/PyMC example | Poisson & NegBin count GLM, splines (Ch 9, 14) |
| **Chimpanzees (prosocial)** | rethinking `chimpanzees.csv` | hierarchical logistic, varying intercepts (Ch 10) |
| **Baseball batting (Efron–Morris / Lahman)** | small hardcoded table or `pybaseball` | shrinkage / partial pooling intuition (Ch 10) |
| **Mauna Loa CO₂** | `statsmodels` co2 or sklearn | Gaussian processes, structural time series (Ch 12, 15) |
| **Airline passengers / `AirPassengers`** | classic CSV | time series, trend+seasonality (Ch 15) |
| **Coal mining disasters** | PyMC built-in `pm.get_data("coal.csv")`-style | change-point / Poisson (Ch 13, 15) |
| **Old Faithful / `geyser`** | seaborn | mixture models, density estimation (Ch 13) |
| **Tips / Diamonds** | seaborn | GLMs, interactions, robust regression (Ch 9) |
| **Rugged / GDP (terrain)** | rethinking `rugged.csv` | interactions, log models (Ch 9) |

When a dataset URL is uncertain, prefer one of: seaborn built-ins, `pm.get_data`, the
`rmcelreath/rethinking` GitHub raw CSVs, or generate **synthetic data with a known
generative process** (excellent for teaching — you can show recovery of true parameters).
Synthetic-with-known-truth is ENCOURAGED for illustrating estimators and diagnostics.

## 8. Canonical resources (every "further reading" draws from these + dossier extras)

Books:
- McElreath, *Statistical Rethinking*, 2nd ed (2020) + free lecture videos + `rethinking` R pkg
  and PyMC port (`pymc-devs/pymc-resources`, "Statistical Rethinking" ports). Cite chapters.
- Martin, Kumar, Lao, *Bayesian Modeling and Computation in Python* (CRC, 2021) —
  free online at **bayesiancomputationbook.com**. The PyMC/ArviZ companion to this course.
- Martin, *Bayesian Analysis with Python*, 3rd ed (2024).
- Gelman, Carlin, Stern, Dunson, Vehtari, Rubin, *Bayesian Data Analysis* (BDA3, 2013) —
  free PDF. The reference.
- Gelman & Hill, *Data Analysis Using Regression and Multilevel/Hierarchical Models* (2007);
  Gelman, Hill & Vehtari, *Regression and Other Stories* (2020, free PDF).
- Kruschke, *Doing Bayesian Data Analysis*, 2nd ed (the "puppy book").

Papers / case studies:
- Gelman et al., **"Bayesian Workflow"** (2020), arXiv:2011.01808.
- Betancourt, **"Towards A Principled Bayesian Workflow"**, "A Conceptual Introduction to
  Hamiltonian Monte Carlo" (arXiv:1701.02434), "Robust Gaussian Processes", "Hierarchical
  Modeling", "Identity Crisis" — all at betanalpha.github.io/writing.
- Vehtari, Gelman, Simpson, Carpenter, Bürkner, **"Rank-normalization, folding, and
  localization: An improved R̂"** (2021).
- Vehtari, Gelman, Gabry, **"Practical Bayesian model evaluation using LOO-CV and WAIC"** (2017)
  + the PSIS paper (Vehtari et al. 2015/2024).
- Talts, Betancourt, Simpson, Vehtari, Gelman, **"Validating Bayesian Inference Algorithms
  with Simulation-Based Calibration"** (2018).
- Gelman, **"Prior distributions for variance parameters in hierarchical models"** (2006);
  Simpson et al., **"Penalising Model Component Complexity"** (PC priors, 2017);
  the **Stan prior-choice wiki** (github.com/stan-dev/stan/wiki/Prior-Choice-Recommendations).
- Hoffman & Gelman, **"The No-U-Turn Sampler"** (2014); Hoffman et al. ADVI (2017);
  Zhang et al. / Pathfinder (2022).
- Neal, **"Slice sampling"**/**"MCMC using Hamiltonian dynamics"** (2011).

Docs / blogs:
- PyMC docs + **example gallery** (pymc.io/projects/examples), PyMC discourse.
- ArviZ docs (python.arviz.org). Bambi docs (bambinos.github.io/bambi).
- Stan User's Guide + Reference Manual (mc-stan.org). NumPyro docs. BlackJAX docs. nutpie.
- Blogs: Thomas Wiecki (twiecki.io), Austin Rochford, PyMC Labs blog, Junpeng Lao,
  Ravin Kumar, "While My MCMC Gently Samples".

## 9. The chapter map (authoritative — cross-reference BY THESE numbers/titles)

Folder: `bayesian_learning/`. Files are `NN-slug.md`.

- **00** `00-overview-and-setup.md` — Course overview, how to study, environment setup,
  the Bayesian workflow at a glance, the master dataset index, notation. [dossier R1, R11]
- **01** `01-foundations-of-bayesian-inference.md` — Probability as extended logic, Bayes'
  theorem, likelihood/prior/posterior/evidence, generative modeling, Bayesian vs frequentist,
  why go Bayesian, the small-world/large-world idea. [R1]
- **02** `02-priors-1-concepts-and-common-priors.md` — What a prior is; flat/improper vs
  proper; conjugate; weakly informative; informative; regularizing; PC priors; the catalog of
  common distributions and exactly when to reach for each; priors on location vs scale. [R2]
- **03** `03-priors-2-choosing-building-and-checking.md` — Prior predictive checks; schools of
  thought (subjective/objective/weakly-informative/empirical Bayes); standardization & scaling;
  priors for scale/variance params (HalfNormal, HalfCauchy, Exponential, Gelman-2006, LKJ);
  choosing hyperparameters; eliciting priors; "should I look at the data first?" debate;
  sensitivity analysis. [R2]
- **04** `04-inference-1-analytic-grid-and-quadratic-approximation.md` — Conjugate/analytic
  posteriors; grid approximation; Laplace / quadratic approximation (QUAP, à la McElreath);
  when each works and fails. [R4]
- **05** `05-inference-2-mcmc-from-metropolis-to-nuts.md` — Markov chains; Metropolis–Hastings;
  Gibbs; curse of dimensionality & the typical set; HMC (Hamiltonian dynamics, leapfrog,
  momentum); NUTS; warmup/adaptation, step size, mass matrix, target_accept; what each knob does. [R3]
- **06** `06-inference-3-variational-inference.md` — KL & the ELBO; mean-field ADVI; full-rank;
  normalizing-flow VI; Pathfinder; VI vs MCMC; where VI lies to you and how to detect it. [R4]
- **07** `07-mcmc-diagnostics-and-debugging.md` — R̂ (rank-normalized), ESS bulk/tail, MCSE,
  trace/rank/forest plots, divergences, energy/BFMI, target_accept tuning, the funnel,
  centered vs non-centered reparameterization (deep), the ArviZ diagnostic loop. [R5]
- **08** `08-the-bayesian-workflow.md` — Betancourt's principled workflow + Gelman et al.
  "Bayesian Workflow": prior predictive → fit → diagnose → posterior predictive checks →
  model expansion/comparison → SBC (simulation-based calibration) → fake-data recovery. [R1, R5]
- **09** `09-glms-and-bambi.md` — Linear regression; logistic; Poisson & Negative Binomial;
  robust (Student-t); ordinal/categorical; link functions & interpretation (log-odds, IRR);
  the Bambi formula API with raw-PyMC equivalents; interactions; contrasts/marginal effects. [R6]
- **10** `10-hierarchical-and-multilevel-models.md` — Partial pooling; varying intercepts &
  slopes; radon flagship; shrinkage; hyperpriors; non-centered parameterization in depth;
  LKJ correlation priors for correlated varying effects; predicting new groups. [R7]
- **11** `11-model-comparison-and-evaluation.md` — Predictive view; ELPD; LOO-CV via PSIS;
  WAIC; k̂ diagnostic; `az.compare`; stacking & model averaging; posterior predictive checks
  for comparison; calibration; the trouble with Bayes factors. [R8]
- **12** `12-gaussian-processes.md` — Kernels/covariance functions; GP regression
  (`pm.gp.Marginal`); latent GPs for non-Gaussian likelihoods (`pm.gp.Latent`);
  priors on lengthscale/amplitude; mean functions; combining kernels; sparse & HSGP
  approximations for scale; the GP↔neural-net & GP↔spline connections. [R9]
- **13** `13-mixture-models-and-latent-variables.md` — Finite mixtures; identifiability &
  label switching (ordered constraint); marginalizing discrete latents (why PyMC marginalizes,
  Stan `log_sum_exp`); zero-inflated & hurdle models; latent-variable thinking. [R10]
- **14** `14-nonparametric-and-flexible-models.md` — Bayesian splines/basis functions;
  BART (`pymc_bart`); Dirichlet-process / infinite mixtures (stick-breaking); when to reach
  for nonparametrics vs GPs vs BART. [R10]
- **15** `15-time-series-and-state-space-models.md` — AR; Gaussian random walk; structural
  time series (trend+seasonality); stochastic volatility; state-space/Kalman view;
  change-points; forecasting with uncertainty. [R11]
- **16** `16-missing-data-the-bayesian-way.md` — MCAR/MAR/MNAR; imputation as parameters;
  PyMC automatic imputation (masked arrays); marginalizing/​summing out missingness;
  modeling the missingness mechanism (selection models); vs multiple imputation; censoring
  & truncation as relatives. [R11]
- **17** `17-scaling-and-advanced-computation.md` — Backends: NumPyro/JAX, nutpie, BlackJAX;
  `pm.sample(nuts_sampler=...)`; GPU; within-chain parallelism (`reduce_sum` in Stan);
  minibatch ADVI; marginalization for speed; sparse/HSGP recap; sparse data & `pm.Data`;
  how to pick a sampler/backend; profiling & common performance traps. [R11]
- **18** `18-capstone-case-studies.md` — Two or three end-to-end case studies that chain the
  WHOLE workflow on real datasets (e.g., a hierarchical count model; a GP time series; a
  logistic model with missing data), explicitly walking every workflow step. [all]
- **A** `appendix-cheatsheets-and-resources.md` — Distribution cheat-sheet; prior cheat-sheet
  (param → recommended prior → why); diagnostic thresholds table (R̂<1.01, ESS, k̂<0.7,
  BFMI, divergences=0); troubleshooting table (error/symptom → cause → fix); a glossary;
  the master resource list; a "what to learn next" roadmap. [all]

## 10. Consistency rules (so chapters don't contradict each other)

- Default sampler call shown as `pm.sample(1000, tune=1000, chains=4, target_accept=0.9,
  random_seed=RANDOM_SEED)`; raise `target_accept` to 0.95–0.99 when discussing divergences.
- Thresholds quoted consistently: **R̂ ≤ 1.01** good (≤1.05 tolerable in early drafts but aim
  1.01); **ESS_bulk & ESS_tail ≳ 400** (≳100 per chain) for stable estimates; **0 divergences**
  is the target; **k̂ < 0.7** for trustworthy PSIS-LOO (≥0.7 unreliable, ≥1.0 bad);
  **BFMI < 0.3** flags energy problems. Define each where first used (Ch 7) and reuse.
- "Non-centered parameterization" is introduced conceptually in Ch 5/7 and used as the default
  cure for funnels in Ch 10 — keep terminology identical.
- Always present **standardization of predictors** as the default for setting sane priors (Ch 3
  establishes it; later chapters assume it).
- Sampling returns **InferenceData**; never call it "trace" except to note the old v3 name once.
- When showing Bambi, ALWAYS also show or reference the equivalent PyMC model.

## 11. Output discipline for writing subagents

- Write ONLY the chapter Markdown to the assigned file path. No preamble to the user.
- Return to the orchestrator a SHORT summary: file path, approx word count, sections covered,
  datasets used, and anything you flagged as uncertain (e.g., an API you weren't 100% sure of).
- Do NOT dump the chapter text back to the orchestrator (keeps context lean).
- If a code API is uncertain, prefer the canonical spelling in §3 and add a brief `# verify in
  current PyMC docs` comment rather than inventing a symbol.
