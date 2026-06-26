# R1 — Foundations + The Bayesian Workflow

Research dossier for chapter writers. Feeds **Ch 00 (overview/setup/how-to-study)**, **Ch 01
(foundations of inference)**, and **Ch 08 (the Bayesian workflow)**. Reference material —
dense, verify-before-you-write. Accuracy over polish.

---

## Scope & which chapters use this

- **Ch 00** — course framing, "how to study," environment setup (install lines), the workflow
  at a glance, master dataset index, notation. Pull the *install lines* and the *workflow
  one-pager* from here.
- **Ch 01** — probability as extended logic (Cox/Jaynes), Bayes' theorem decomposition,
  generative modeling, small-world/large-world (McElreath), Bayesian vs frequentist,
  why-Bayesian arguments. Pull the *conceptual core* and *interval-interpretation* material.
- **Ch 08** — the *full* workflow: Gelman et al. (2020) AND Betancourt's principled workflow,
  reconciled; prior predictive → fit → diagnose → posterior predictive → expand/compare → SBC
  → fake-data recovery **as concepts** (mechanics live in R3/R5/R8). Pull the *two enumerated
  workflows and the reconciliation table* from here.

Boundary: this dossier owns the *concepts and narrative*. It does NOT own: MCMC/NUTS internals
(R3), diagnostics thresholds/divergences (R5), priors-in-depth (R2), LOO/WAIC mechanics (R8).
Cross-reference, don't duplicate.

---

## Verified API & idioms (confirmed against current PyMC v5 docs)

> ⚠️ **FROZEN-STACK ALERT — read first.** As of mid-2026, **PyMC 6.0.x exists** (6.0.0 released
> 2026-05-13) and the official install page now defaults to `"pymc>=6"`. **This course is frozen
> to PyMC v5** per the bible. The last v5 series is **5.28.5** (2026-05-01). So the course MUST
> pin to v5 explicitly and NOT copy the docs' default `pymc>=6` line. Use `"pymc>=5.16,<6"` (any
> recent v5 is fine). All idioms below are verified v5 and are also stable in current v5 releases.

**Prior predictive** — uses `draws=` (NOT `samples=`), stores into `idata.prior`:
```python
import pymc as pm
RANDOM_SEED = 8927
with model:
    idata = pm.sample_prior_predictive(draws=500, random_seed=RANDOM_SEED)
# samples live in idata.prior and idata.prior_predictive
```

**Sampling** — returns `az.InferenceData`; posterior lands in `idata.posterior`:
```python
with model:
    idata = pm.sample(1000, tune=1000, chains=4, target_accept=0.9,
                      random_seed=RANDOM_SEED)
```

**Posterior predictive** — `extend_inferencedata=True` appends in place; stores into
`idata.posterior_predictive` (or `idata.predictions` when `predictions=True`):
```python
with model:
    pm.sample_posterior_predictive(idata, extend_inferencedata=True,
                                   random_seed=RANDOM_SEED)
# out-of-sample / new-data predictions:
with model:
    pm.sample_posterior_predictive(idata, var_names=["y_obs"],
                                   predictions=True, extend_inferencedata=True,
                                   random_seed=RANDOM_SEED)
```

**Model skeleton with named dims/coords** (the bible's required style):
```python
coords = {"obs_id": df.index}
with pm.Model(coords=coords) as model:
    x = pm.Data("x", df["x"].values, dims="obs_id")      # mutable in v5; swap for prediction
    a = pm.Normal("a", 0, 1)
    b = pm.Normal("b", 0, 1)
    sigma = pm.HalfNormal("sigma", 1)
    mu = pm.Deterministic("mu", a + b * x, dims="obs_id")
    y = pm.Normal("y_obs", mu=mu, sigma=sigma,
                  observed=df["y"].values, dims="obs_id")
```
- `pm.Data` is **mutable by default in v5** (the v4 `pm.MutableData`/`pm.ConstantData` split was
  removed; `pm.MutableData` may still alias but prefer `pm.Data`). Swap with
  `pm.set_data({"x": new_x})` then re-run `sample_posterior_predictive` for predictions.
- `pm.sample()` auto-selects NUTS for continuous params. Backends:
  `pm.sample(nuts_sampler="numpyro" | "nutpie" | "blackjax")`.

**ArviZ entry points the workflow chapter names** (mechanics in R5/R8, but spellings to use):
`az.summary(idata)`, `az.plot_trace`, `az.plot_ppc(idata)` (PPC overlay),
`az.plot_bpv(idata)` (Bayesian p-value / calibration), `az.plot_dist_comparison`,
`az.loo(idata)`, `az.compare({...})`, `az.plot_forest`, `az.plot_posterior`.
Prior predictive is plotted with `az.plot_ppc(idata, group="prior")`.

> 🔎 **Verified.** All the above signatures (`draws=`, `extend_inferencedata=`, `predictions=`,
> group names `prior`/`posterior_predictive`/`predictions`) are confirmed against the current
> PyMC "Prior and Posterior Predictive Checks" core notebook. `az.plot_bpv` exists in ArviZ but
> double-check the exact kwargs (`kind="p_value"|"u_value"`) against the version installed.

---

## Key concepts & equations the chapter must nail

**1. Probability as extended logic (Cox / Jaynes).** Cox's theorem: any system of "plausibility"
that is (a) represented by real numbers, (b) qualitatively consistent with Boolean logic, and
(c) internally consistent, is **isomorphic to probability theory**. Hence probability is the
unique calculus of rational degrees of belief — not merely long-run frequency. Jaynes,
*Probability Theory: The Logic of Science* (2003), is the canonical source; ch. 1–2 derive the
sum and product rules from desiderata. This is the philosophical bedrock of "why Bayesian."

**2. Bayes' theorem decomposition.** For data $y$ and parameters $\theta$:
$$ p(\theta \mid y) = \frac{p(y \mid \theta)\,p(\theta)}{p(y)}, \qquad
   p(y) = \int p(y\mid\theta)\,p(\theta)\,d\theta. $$
- $p(\theta)$ **prior**, $p(y\mid\theta)$ **likelihood** (a function of $\theta$, NOT a
  distribution over $\theta$), $p(\theta\mid y)$ **posterior**, $p(y)$ **evidence / marginal
  likelihood** (a normalizing constant for inference; the hard integral).
- Proportional form (what you actually compute): $p(\theta\mid y) \propto p(y\mid\theta)\,p(\theta)$.
- Posterior predictive: $p(\tilde y \mid y) = \int p(\tilde y\mid\theta)\,p(\theta\mid y)\,d\theta$.
- Prior predictive (a.k.a. marginal/evidence over the prior): $p(\tilde y) = \int p(\tilde
  y\mid\theta)\,p(\theta)\,d\theta$. **Prior predictive checks sample from this.**
- ASCII for skimmers: `posterior ∝ likelihood × prior`; `evidence = ∫ likelihood × prior dθ`.

**3. Generative modeling.** A Bayesian model is a *joint distribution* $p(y,\theta) =
p(y\mid\theta)\,p(\theta)$ — a recipe to **simulate fake data**. You can run it forward (sample
$\theta\sim$prior, then $y\sim$likelihood) to get prior predictive draws, or backward (condition
on observed $y$) to get the posterior. "Thinking generatively — how could this data have arisen?"
is the design heuristic (McElreath). This is the conceptual seed of prior predictive checks,
fake-data simulation, and SBC.

**4. Small world vs large world (McElreath, *Statistical Rethinking* 2e, Ch 1–2).**
- **Golem**: a model is a brainless golem that does exactly what its assumptions say — powerful,
  literal, no wisdom of its own (Ch 1, "The Golem of Prague").
- **Small world**: the self-contained logical world *inside* the model, where every possibility
  is enumerated. Bayesian updating is *provably optimal* here.
- **Large world**: messy reality where the model is deployed. Optimality in the small world gives
  **no guarantee** in the large world; this is why model checking (PPC) and expansion exist.
- Use the metaphor verbatim — it's load-bearing for Ch 01 and motivates the whole workflow.

**5. Bayesian vs frequentist — what is *actually* different.**
- **What's random.** Frequentist: $\theta$ is a *fixed unknown constant*; the *data* (and any
  interval computed from it) are random. Bayesian: $\theta$ has a *probability distribution*
  expressing uncertainty; the data, once observed, are fixed.
- **Confidence interval (95%)**: a property of the *procedure* — under indefinite resampling, 95%
  of computed intervals would cover the true $\theta$. A *given* CI either covers or doesn't;
  "95%" is NOT the probability this interval contains $\theta$. (Classic misinterpretation trap.)
- **Credible interval (95%)**: given this data and prior, $P(\theta \in [\ell,u]\mid y)=0.95$ —
  a direct probability statement about $\theta$. This is what people *wish* a CI meant.
- **When it matters in practice**: small samples / weak data (prior actually contributes);
  hierarchical structure (partial pooling vs no/complete pooling); decision-making under
  uncertainty (full posterior → expected utility); propagating uncertainty into predictions;
  sequential updating; nuisance parameters (Bayes integrates them out, no plug-in). With huge
  $n$ and flat priors the two often numerically coincide (Bernstein–von Mises) — say so honestly.
- **Likelihood principle**: Bayesian inference depends on data only through the likelihood; it is
  unaffected by the experimenter's stopping intention (unlike p-values). A genuine philosophical
  difference, not just bookkeeping.

**6. Why go Bayesian (the honest argument).** (i) Coherent uncertainty quantification — full
posterior, not a point + SE; (ii) priors let you *use* domain knowledge and *regularize*
(shrinkage falls out for free, e.g. hierarchical models); (iii) generative thinking → checkable
models; (iv) uniform treatment of complex models (hierarchies, latent variables, missing data)
without bespoke estimators; (v) honest propagation of uncertainty into predictions/decisions.
**Costs to admit**: computation (MCMC is slower, can fail), prior elicitation effort, and you
must actually *check* the model. Don't oversell — the bible's voice is honest mentorship.

**7. The four checking concepts (defined here, mechanics elsewhere).**
- **Prior predictive check**: simulate $\tilde y \sim p(\tilde y)$ *before* seeing data; are the
  implied datasets physically plausible? Catches absurd priors (R2 owns prior design).
- **Posterior predictive check (PPC)**: simulate $\tilde y \sim p(\tilde y\mid y)$ and compare to
  observed $y$ via test quantities/plots; does the fitted model reproduce features of the data?
- **Fake-data / parameter recovery**: simulate one dataset from known $\theta^\*$, fit, check the
  posterior covers $\theta^\*$. Cheapest sanity check that code + model are sane.
- **Simulation-based calibration (SBC)** (Talts et al. 2018): the rigorous generalization — for
  many draws $\theta^{(i)}\sim p(\theta)$, simulate $y^{(i)}$, fit, and compute the **rank** of
  $\theta^{(i)}$ within its posterior draws. If inference is correct, ranks are **Uniform**;
  deviations (∪/∩/skew shapes in the rank histogram) diagnose specific failures. R5 owns details.

---

## The two workflows — enumerated and reconciled (the heart of Ch 08)

### A. Gelman et al. (2020), "Bayesian Workflow" — arXiv:2011.01808 (77 pp.). Section map (verified from the paper):
1. **Introduction** — 1.1 From Bayesian inference to Bayesian workflow; 1.2 Why do we need a
   workflow; 1.4 Organizing the many aspects; 1.5 Aim/structure.
2. **Before fitting a model** — 2.1 Choosing an initial model; 2.2 Modular construction;
   2.3 Scaling/transforming parameters; **2.4 Prior predictive checking**; 2.5 Generative and
   partially generative models.
3. **Fitting a model** — 3.1 Initial values, adaptation, warmup; 3.2 How long to run; 3.3
   Approximate algorithms/models; **3.4 Fit fast, fail fast**.
4. **Using constructed/simulated data to find and understand problems** — **4.1 Fake-data
   simulation**; **4.2 Simulation-based calibration**; 4.3 Experimentation using constructed data.
5. **Addressing computational problems** — 5.1 The folk theorem of statistical computing
   (*computational trouble usually signals a model problem*); 5.2 Start simple & complex, meet in
   the middle; 5.3 Long-running fits; 5.4 Monitoring intermediate quantities; 5.5 Stacking to
   reweight poorly-mixing chains; 5.6 Multimodality & difficult geometry; **5.7 Reparameterization**;
   5.8 Marginalization; 5.9 Adding prior information; 5.10 Adding data.
6. **Evaluating and using a fitted model** — **6.1 Posterior predictive checking**; 6.2
   Cross-validation & influence of points; 6.3 Influence of prior information; 6.4 Summarizing
   inference & propagating uncertainty.
7. **Modifying a model** — 7.1 Constructing a model for the data; 7.2 Incorporating additional
   data; 7.3 Working with priors; **7.4 A topology of models** (the space of models you traverse).
8. **Understanding & comparing multiple models** — 8.1 Visualizing models in relation to each
   other; 8.2 Cross-validation & model averaging; 8.3 Comparing many models.
9. **Modeling as software development** — version control, testing, reproducibility, readability.
10. **Example: model building & expansion — Golf putting** (the famous worked example: logistic →
    first-principles geometry → new data → physics with "how hard the ball is hit" → fudge factor).
11. **Example: unexpected multimodality — Planetary motion.**
12. **Discussion** — iterative model building, model selection/overfitting, "bigger datasets
    demand bigger models," prediction/generalization/poststratification.

> 💡 **Gelman's stance**: workflow is *iterative and many-model* — you fit a sequence/topology of
> models, not one. "The folk theorem" (5.1) is quotable: bad sampling usually means a bad model.

### B. Betancourt, "Towards A Principled Bayesian Workflow" (betanalpha.github.io case study). Verified structure from the source `.Rmd`:

**Four evaluating questions** (exact titles):
- **Question One: Domain Expertise Consistency** — are the model's assumptions consistent with
  domain knowledge? (answered by prior pushforward / prior predictive checks)
- **Question Two: Computational Faithfulness** — can our algorithm actually fit the model? (SBC,
  diagnostics)
- **Question Three: Inferential Adequacy** — will inferences be good enough for the application?
  (posterior predictive sensitivity, do we have enough information?)
- **Question Four: Model Adequacy** — does the model capture the true data-generating structure?
  (posterior retrodictive checks against observed data)

**Three phases** with **fifteen steps** (exact step titles, verified):
- *Phase — Pre-Model, Pre-Data*: **1 Conceptual Analysis** · **2 Define Observational Space** ·
  **3 Construct Summary Statistics**.
- *Phase — Post-Model, Pre-Data*: **4 Model Development** · **5 Construct Summary Functions** ·
  **6 Simulate Bayesian Ensemble** · **7 Prior Checks** · **8 Configure Algorithm** ·
  **9 Fit Simulated Ensemble** · **10 Algorithmic Calibration** (SBC over the ensemble) ·
  **11 Inferential Calibration**.
- *Phase — Post-Model, Post-Data*: **12 Fit the Observation** · **13 Diagnose Posterior Fit** ·
  **14 Posterior Retrodictive Checks** · **15 Celebrate**.

> 📜 Betancourt calls successful checks "retrodictive" (the model *retrodicts* the observed data)
> rather than "predictive," to stress you're checking against data already in hand. Use his term.

### C. Reconciliation (build this table in Ch 08)

| Concept | Gelman et al. (2020) | Betancourt (principled) |
|---|---|---|
| Frame the problem | §2.1 choosing initial model | Step 1 Conceptual Analysis; Q1 |
| What can we observe | §2.5 generative models | Steps 2–3 Observational space + summary stats |
| Build the model | §2.2 modular; §2.3 scaling | Step 4 Model Development |
| Check priors *before* data | §2.4 prior predictive checking | Steps 6–7 Simulate ensemble + Prior Checks; Q1 |
| Can we even fit it? | §3 fitting; §3.4 fail fast | Step 8 Configure Algorithm; Q2 |
| Validate the *algorithm* | §4.1 fake data; §4.2 SBC | Steps 9–10 Fit ensemble + Algorithmic Calibration; Q2 |
| Will inference be useful? | §6.4 summarize/propagate | Step 11 Inferential Calibration; Q3 |
| Fit the real data | §3 | Step 12 Fit the Observation |
| Diagnose computation | §5 (folk theorem, reparam) | Step 13 Diagnose Posterior Fit |
| Check fit to data | §6.1 posterior predictive checking | Step 14 Posterior Retrodictive Checks; Q4 |
| Improve / compare | §7 modifying; §8 comparing | (loop back; expand model) |
| Engineering hygiene | §9 modeling as software | (implicit) |

**Key difference to teach**: Gelman is *descriptive & iterative* ("here's the messy reality of
fitting many models, and a toolbox"); Betancourt is *prescriptive & sequential* ("here is a
disciplined checklist; calibrate the algorithm *before* you ever touch real data"). They agree on
the spine: **generative model → prior predictive check → fit → computational diagnostics →
posterior predictive check → expand**. SBC/fake-data sit between "can I fit it" and "fit real
data." Present the spine first, then both authors as two lenses.

---

## Common errors → causes → fixes (foundations/setup-level; modeling errors live in R5)

| Symptom | Cause | Fix |
|---|---|---|
| `pip install pymc` pulls **PyMC 6** | docs now default to v6; course is frozen v5 | Pin: `conda create -c conda-forge -n pymc_env "pymc>=5.16,<6"` or `pip install "pymc>=5.16,<6"` |
| `AttributeError: pm.MutableData` / `pm.sample_ppc` / `pm.Model().trace` | v3/v4 idioms | Use `pm.Data` (mutable), `pm.sample_posterior_predictive`, and `idata` (InferenceData), not "trace" |
| Slow/broken C compile, BLAS warnings on install | PyMC uses PyTensor → needs a C toolchain / `g++` | Prefer **conda-forge** (ships compilers + mkl). On Mac install Xcode CLT; the conda route avoids most pain |
| `sample_prior_predictive(samples=...)` errors | arg is `draws=` not `samples=` | Use `draws=` |
| Prior predictive draws look insane (e.g. heights of 10^6 cm) | unstandardized predictors + vague priors → small-world nonsense | Standardize predictors (R2/Ch 03), use weakly-informative priors, re-check prior predictive |
| "My 95% CI means 95% chance θ is inside" stated for a *frequentist* CI | conflating CI with credible interval | Teach the fixed-vs-random distinction explicitly; only the credible interval licenses that statement |
| NumPyro/JAX backend import fails | optional dep not installed | `pip install "numpyro[cpu]"` (or jax) separately; it's optional |
| nutpie not found | optional, needs Rust-compiled wheel | `conda install -c conda-forge nutpie` or `pip install nutpie` |
| CmdStanPy `CmdStanModel` can't find Stan | CmdStan toolchain not installed | `pip install cmdstanpy` then `python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"` |

---

## Datasets & load snippets (for the first worked example in Ch 01/08)

The bible's flagship for first models is **Howell !Kung**. Verified-style load (rethinking repo
raw CSV, semicolon-separated):
```python
import pandas as pd
url = "https://raw.githubusercontent.com/rmcelreath/rethinking/master/data/Howell1.csv"
d = pd.read_csv(url, sep=";")
adults = d[d.age >= 18].copy()   # height ~ weight is linear only for adults
```
**Eight Schools** (hardcoded — canonical hierarchical/funnel teaching set, good for "why
Bayesian / partial pooling" foreshadowing in Ch 01):
```python
import numpy as np
J = 8
y = np.array([28, 8, -3, 7, -1, 1, 18, 12])
sigma = np.array([15, 10, 16, 11, 9, 11, 10, 18])
```
**Synthetic with known truth** (bible-encouraged for foundations — show parameter recovery &
prior/posterior predictive on a process you control):
```python
rng = np.random.default_rng(8927)
a_true, b_true, s_true = 2.0, 0.5, 1.0
x = rng.normal(size=100)
y = rng.normal(a_true + b_true * x, s_true)
```
For PyMC built-ins later chapters use, mention `pm.get_data("radon.csv")`-style loaders exist
(verify the exact filename in current docs before asserting). Eight-schools also ships in ArviZ:
`az.load_arviz_data("centered_eight")` / `"non_centered_eight"` — handy InferenceData for showing
diagnostics without sampling.

---

## Curated resources (each: real URL + why + pointer)

- **Gelman, Vehtari, Simpson, Margossian, Carpenter, Yao, Kennedy, Gabry, Bürkner, Modrák,
  "Bayesian Workflow" (2020)** — https://arxiv.org/abs/2011.01808 — THE workflow paper; use the
  §2–§8 map above. Golf-putting (§10) is the best "model expansion" narrative; §5.1 folk theorem.
- **Betancourt, "Towards A Principled Bayesian Workflow"** —
  https://betanalpha.github.io/assets/case_studies/principled_bayesian_workflow.html — the 4
  questions + 15 steps + 3 phases; rigorous, prescriptive. Source `.Rmd` (for exact step titles):
  https://github.com/betanalpha/knitr_case_studies/tree/master/principled_bayesian_workflow
- **Betancourt writing index** — https://betanalpha.github.io/writing/ — gateway to all case
  studies (HMC, hierarchical, identifiability) referenced across the course.
- **McElreath, *Statistical Rethinking* 2e** — book page
  https://xcelab.net/rm/ — Ch 1 (Golem of Prague), Ch 2 (Small Worlds & Large Worlds) are the
  source for golem / small-world / generative framing. Free lectures:
  https://github.com/rmcelreath/stat_rethinking_2023
- **PyMC port of Statistical Rethinking** — https://github.com/pymc-devs/pymc-resources — runnable
  PyMC v5 versions of McElreath's models; mine for Ch 01 code.
- **Martin, Kumar, Lao, *Bayesian Modeling and Computation in Python* (2021)** —
  https://bayesiancomputationbook.com — the PyMC/ArviZ companion; Ch 1 (Bayesian inference) and
  Ch 9 (workflow / "end-to-end") parallel this dossier directly.
- **Gelman, Hill, Vehtari, *Regression and Other Stories* (2020, free PDF)** —
  https://avehtari.github.io/ROS-Examples/ — Ch 9 "Prediction and Bayesian inference" and the
  workflow framing; excellent for the Bayesian-vs-frequentist interval discussion.
- **BDA3 (Gelman et al. 2013, free PDF)** — http://www.stat.columbia.edu/~gelman/book/ — the
  reference; Ch 1 probability/Bayes, Ch 6 model checking (posterior predictive), Ch 7 evaluation.
- **Talts, Betancourt, Simpson, Vehtari, Gelman, "Validating Bayesian Inference Algorithms with
  Simulation-Based Calibration" (2018)** — https://arxiv.org/abs/1804.06788 — SBC definition; the
  rank-statistic-is-uniform result and the shape→diagnosis catalog (mechanics in R5).
- **PyMC "Prior and Posterior Predictive Checks" core notebook** —
  https://www.pymc.io/projects/docs/en/latest/learn/core_notebooks/posterior_predictive.html —
  verified v5 idioms for `sample_prior_predictive(draws=...)` /
  `sample_posterior_predictive(..., extend_inferencedata=True, predictions=...)` and group names.
- **PyMC Installation page** — https://www.pymc.io/projects/docs/en/latest/installation.html —
  conda/mamba + optional backends; ⚠️ it now defaults to `pymc>=6` — pin to v5 in the course.
- **PyMC Example Gallery** — https://www.pymc.io/projects/examples/en/latest/gallery.html — and
  GLM linear-regression core notebook
  https://www.pymc.io/projects/docs/en/latest/learn/core_notebooks/GLM_linear.html — clean
  end-to-end (prior predictive → sample → PPC) template for the Ch 08 worked example.
- **ArviZ docs** — https://python.arviz.org/ — `plot_ppc`, `plot_bpv`, `summary`, InferenceData
  schema; the diagnostics vocabulary Ch 08 names.
- **Bambi docs** — https://bambinos.github.io/bambi/ — formula API for the GLM chapter; mention in
  Ch 00's stack tour.
- **Jaynes, *Probability Theory: The Logic of Science* (2003)** — ch. 1–2 for Cox's theorem /
  probability-as-extended-logic; the philosophical anchor for "why Bayesian."
- **Wikipedia "Credible interval"** — https://en.wikipedia.org/wiki/Credible_interval — concise,
  correct fixed-vs-random framing for the CI-vs-credible-interval box (cross-check phrasing).

---

## Environment setup (for Ch 00 — give learners copy-paste lines)

**Recommended (conda-forge, pinned to v5):**
```bash
conda create -c conda-forge -n bayes "pymc>=5.16,<6" arviz bambi numpyro nutpie cmdstanpy
conda activate bayes
pip install pymc-bart            # BART extension (pip is fine)
python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"   # one-time: fetch CmdStan toolchain
# blackjax (optional JAX sampler):
pip install blackjax
```
**pip-only alternative** (needs a working C/C++ compiler for PyTensor):
```bash
pip install "pymc>=5.16,<6" arviz bambi "numpyro[cpu]" nutpie cmdstanpy pymc-bart blackjax
```
Sanity check:
```python
import pymc as pm, arviz as az, bambi as bmb
print(pm.__version__)            # expect 5.x  (NOT 6.x — course is frozen to v5)
```
> 🔧 conda-forge is strongly preferred because it ships the compiler toolchain + MKL/BLAS and
> avoids the most common install failures. NumPyro, nutpie, BlackJAX are *optional* sampler
> backends; CmdStan must be fetched once via `install_cmdstan()`.

---

## "How to study" framing (for Ch 00)

- **Run everything.** This is a practice-first course; type the code, change a prior, break it,
  re-run. The golem only teaches you when you watch it move.
- **Internalize the spine, not 15 steps.** Memorize the workflow *spine* (generative model →
  prior predictive → fit → diagnose → posterior predictive → expand) and treat Gelman's toolbox
  and Betancourt's 15 steps as two zoom levels on it.
- **Always check before you trust.** No posterior is believed until prior predictive looks sane,
  diagnostics are clean (R̂≤1.01, 0 divergences — defined in R5/Ch 07), and PPC matches the data.
- **Generative-first thinking.** For every problem ask "how could this data have arisen?" and
  write the simulator first; the model usually falls out of the simulator.
- **Honest about costs.** Bayesian methods buy coherence and uncertainty at the price of compute
  and discipline. Teach the trade-off, not a sales pitch.

---

## Uncertainties / things the writer should double-check

- **PyMC version**: confirm the exact v5 release the course environment pins (5.28.5 is the last
  v5; any recent v5 is fine). Do NOT use the docs' default `pymc>=6` line — the course is frozen
  to v5. If the course later decides to migrate to v6, this dossier's pin must change.
- **`pm.Data` mutability**: verified mutable-by-default in v5, but confirm `pm.MutableData` is
  still importable (it is aliased in recent v5) before relying on either spelling.
- **`az.plot_bpv` kwargs**: function exists; confirm `kind=`/`t_stat=` options against the
  installed ArviZ version before writing exact calls.
- **`pm.get_data("radon.csv")` / built-in dataset filenames**: verify exact filenames in current
  PyMC before asserting them; the rethinking raw-CSV and ArviZ `load_arviz_data` routes are safer.
- **Gelman §1.3**: the TOC grep skipped a 1.3 subsection number; the §1.x subsection titles above
  are verified for 1.1/1.2/1.4/1.5 — confirm 1.3 if you cite it specifically.
- **Betancourt step *wording*** is verified from the `.Rmd`; the per-step *prose* should be read
  from the rendered case study before paraphrasing in depth.
