# The Bayesian Learning Course
### From first principles to state-of-the-art practice — priors, inference engines, the model zoo, and the full Bayesian workflow, in Python

> *A note from me to you:* this course exists to make you genuinely **expert** at Bayesian
> modeling — not just able to call `pm.sample()`, but able to choose priors deliberately,
> pick and tune the right inference algorithm, read every diagnostic, debug the hard
> failures, and ship the right model for the data in front of you. We go all the way in:
> the math, the intuition, runnable **PyMC v5 / ArviZ / Bambi / Stan / NumPyro** code, the
> errors you *will* hit and exactly how to fix them. Read a chapter, run its code, do its
> exercises, then move on. By the end there is nothing in a working Bayesian's day you
> won't recognize.

This is a self-contained, book-length course (~210k words across 20 chapters). It is
modeled on the way the best teachers in this field work — **Richard McElreath's** intuition
and generative thinking, **Michael Betancourt's** rigor and principled workflow, **Osvaldo
Martin / Ravin Kumar / Junpeng Lao's** practical PyMC + ArviZ craft.

---

## How to use this course

1. **Start with [Chapter 00 — Overview & Setup](00-overview-and-setup.md).** It installs the
   environment, gives you the master dataset index, and shows the workflow at a glance.
2. **Read in order** for a first pass — each chapter builds on the last. The inference
   chapters (04–07) are the engine room; do not skip them.
3. **Run everything.** Theory tells you *why*; the code shows you *that it's true*. Every
   chapter has at least one fully worked, end-to-end example on a real dataset.
4. **Do the exercises** at the end of each chapter, and keep the **[Appendix](appendix-cheatsheets-and-resources.md)**
   open as a quick reference (distribution & prior cheat-sheets, diagnostic thresholds, a
   master troubleshooting table).

> 💡 For every model you learn, force yourself to answer seven questions: *what's the
> generative story? what priors? what likelihood? how do I fit it? how do I diagnose it?
> when does it break? which dataset proves it?* If you can answer those for GLMs,
> hierarchical models, GPs, mixtures, and BART, you are already ahead of most practitioners.

---

## Table of contents

### Part I — Foundations
| # | Chapter | What it teaches |
|---|---|---|
| 00 | [Overview & Setup](00-overview-and-setup.md) | how to study, environment, the workflow at a glance, dataset index |
| 01 | [Foundations of Bayesian Inference](01-foundations-of-bayesian-inference.md) | Bayes' theorem, likelihood/prior/posterior/evidence, generative modeling, Bayesian vs frequentist |

### Part II — Priors
| # | Chapter | What it teaches |
|---|---|---|
| 02 | [Priors I — Concepts & the Common-Prior Catalog](02-priors-1-concepts-and-common-priors.md) | flat/conjugate/weakly-informative/regularizing priors; the full distribution catalog and when to use each |
| 03 | [Priors II — Choosing, Building & Checking](03-priors-2-choosing-building-and-checking.md) | prior predictive checks, scale-parameter priors, standardization, schools of thought, sensitivity analysis |

### Part III — Inference engines
| # | Chapter | What it teaches |
|---|---|---|
| 04 | [Analytic, Grid & Quadratic (Laplace) Approximation](04-inference-1-analytic-grid-and-quadratic-approximation.md) | conjugacy, grid approximation, QUAP/Laplace — and when each fails |
| 05 | [MCMC — from Metropolis to NUTS](05-inference-2-mcmc-from-metropolis-to-nuts.md) | Metropolis, Gibbs, the typical set, HMC, leapfrog, NUTS, warmup & every `pm.sample` knob |
| 06 | [Variational Inference](06-inference-3-variational-inference.md) | KL & the ELBO, ADVI, full-rank, Pathfinder, when VI lies and how to catch it |
| 07 | [MCMC Diagnostics & Debugging](07-mcmc-diagnostics-and-debugging.md) | R̂, ESS, divergences, energy/BFMI, the funnel, centered vs non-centered reparameterization |

### Part IV — The workflow
| # | Chapter | What it teaches |
|---|---|---|
| 08 | [The Bayesian Workflow](08-the-bayesian-workflow.md) | the principled loop, fake-data recovery, prior/posterior predictive checks, simulation-based calibration |

### Part V — The model zoo
| # | Chapter | What it teaches |
|---|---|---|
| 09 | [GLMs & Bambi](09-glms-and-bambi.md) | linear, logistic, Poisson/NegBin, robust, ordinal; link functions; the Bambi formula API with raw-PyMC equivalents |
| 10 | [Hierarchical & Multilevel Models](10-hierarchical-and-multilevel-models.md) | partial pooling, varying slopes, LKJ, the radon flagship, non-centering as the default cure |
| 11 | [Model Comparison & Evaluation](11-model-comparison-and-evaluation.md) | ELPD, PSIS-LOO, WAIC, k̂, stacking, calibration, and why Bayes factors are fragile |
| 12 | [Gaussian Processes](12-gaussian-processes.md) | kernels, GP regression & latent GPs, lengthscale priors, HSGP for scale, the CO₂ showcase |
| 13 | [Mixture Models & Latent Variables](13-mixture-models-and-latent-variables.md) | finite mixtures, label switching & the ordered fix, marginalizing discretes, zero-inflation |
| 14 | [Nonparametric & Flexible Models](14-nonparametric-and-flexible-models.md) | Bayesian splines, BART, Dirichlet-process mixtures — and how to choose between them |
| 15 | [Time Series & State-Space Models](15-time-series-and-state-space-models.md) | AR, random walks, structural time series, stochastic volatility, change-points, forecasting |
| 16 | [Missing Data, the Bayesian Way](16-missing-data-the-bayesian-way.md) | MCAR/MAR/MNAR, imputation as parameters, automatic imputation, censoring & truncation |

### Part VI — Production & mastery
| # | Chapter | What it teaches |
|---|---|---|
| 17 | [Scaling & Advanced Computation](17-scaling-and-advanced-computation.md) | NumPyro/nutpie/BlackJAX backends, GPU, within-chain parallelism, minibatch ADVI, picking a sampler |
| 18 | [Capstone Case Studies](18-capstone-case-studies.md) | three full end-to-end builds walking every workflow step |
| A | [Appendix — Cheat-Sheets & Resources](appendix-cheatsheets-and-resources.md) | distribution & prior cheat-sheets, diagnostic thresholds, master troubleshooting table, glossary, resources |

---

## The recommended path

If you're starting cold: **00 → 01 → 02 → 03**, then the engine room **04 → 05 → 06 → 07**
(don't rush these — they're what separate experts from button-pushers), then **08** to tie
inference into a method. From there the model zoo **09 → 10 → 11** is the practical core most
real work lives in; add **12–16** as your problems demand (GPs, mixtures, nonparametrics,
time series, missing data). Finish with **17** (make it fast) and **18** (put it all together).
Keep the **Appendix** open throughout.

## The stack

**PyMC v5** for modeling · **ArviZ** for all diagnostics & plots · **Bambi** for fast
formula-style GLMs (always shown alongside the raw-PyMC model) · **Stan via CmdStanPy** for a
second perspective and the things Stan does best · **NumPyro / nutpie / BlackJAX** for scaling
and GPU. Setup instructions are in [Chapter 00](00-overview-and-setup.md).

---

*Built as a comprehensive, one-shot Bayesian curriculum. If you find an error or a stale API,
it's a feature request: Bayesian software moves fast, so verify any flagged-uncertain call
against the current docs — every chapter points you to them.*
