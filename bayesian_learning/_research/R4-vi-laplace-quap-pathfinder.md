# R4 — Approximate Inference: Grid, Laplace/Quadratic (QUAP) + Variational + Pathfinder

> Research dossier for the chapter writers. Reference material, dense and technical.
> Feeds **Ch 04** (`04-inference-1-analytic-grid-and-quadratic-approximation.md`) and
> **Ch 06** (`06-inference-3-variational-inference.md`). Companion dossiers: R3 (MCMC/NUTS,
> Ch 05) and R5 (diagnostics, Ch 07) — VI/Laplace are validated *against* NUTS, so cross-ref.

---

## Scope & which chapters use this

- **Ch 04** uses: grid approximation; Laplace / quadratic approximation (QUAP, McElreath's
  `quap`); the Gaussian-at-the-mode idea; when valid / when it breaks. Also the *analytic/conjugate*
  thread lives in Ch 04 but is mostly R1/R2 territory — here we cover the **approximation** methods.
- **Ch 06** uses: KL divergence + ELBO derivation; mean-field vs full-rank ADVI (Kucukelbir et al.
  2017); SVGD/ASVGD; normalizing-flow VI (concept); Pathfinder (Zhang et al. 2022); PyMC `pm.fit`
  and `pymc_extras` API; *how VI lies* (variance underestimation, mode-seeking) and *how to detect it*
  (compare to NUTS, PSIS k̂ for VI per Yao et al. 2018).

The narrative arc the bible implies: grid (exact-ish, dies in >2–3 dims) → Laplace/QUAP (cheap,
Gaussian, McElreath's training wheels) → MCMC/NUTS (Ch 05, the gold standard) → VI (Ch 06, fast
but biased) → Pathfinder (the modern "best of both" for initialization / fast approx).

---

## Verified API & idioms (confirmed against current docs)

### Grid approximation (pure NumPy — no special API)
Standard teaching pattern (McElreath SR §2.4.3, "globe tossing"):
```python
import numpy as np
from scipy import stats

grid = np.linspace(0, 1, 1000)              # parameter grid (here a probability p)
prior = np.ones_like(grid)                  # flat prior (or stats.norm.pdf(grid, ...))
likelihood = stats.binom.pmf(6, n=9, p=grid)  # 6 successes in 9 tosses
unnorm_post = likelihood * prior
posterior = unnorm_post / np.trapz(unnorm_post, grid)   # normalize; np.trapezoid in newer numpy
# draw posterior samples by resampling the grid with replacement, weighted by posterior:
samples = np.random.choice(grid, size=10_000, p=posterior/posterior.sum())
```
Key point: grid cost is `points**n_params` → **curse of dimensionality**. Fine for 1–2 params,
hopeless past ~3–4. This is the pedagogical motivation for everything that follows.

### Laplace / quadratic approximation in PyMC v5 — the CURRENT recommended way

`quap` is a **`rethinking` (R) concept**, NOT a PyMC function. There is no `pm.quap`. The PyMC-native
equivalents, in order of preference (all VERIFIED against current docs, June 2026):

**(1) `pymc_extras.fit_laplace` — the recommended one-call API.** Confirmed signature (pymc_extras
0.9.x/0.10.x):
```python
from pymc_extras.inference import fit_laplace   # or: import pymc_extras as pmx; pmx.fit_laplace(...)

with model:
    idata = fit_laplace(
        optimize_method="BFGS",   # default; any scipy.optimize method
        draws=500,                # samples drawn from the fitted Gaussian (default 500)
        chains=None,              # cosmetic chain dim
        random_seed=RANDOM_SEED,
        include_transformed=True, # report on unconstrained scale too
    )                            # -> returns an arviz DataTree / InferenceData-like object
```
It finds the MAP, computes the inverse Hessian (covariance) at the mode, builds a MvNormal, and
returns posterior draws as a DataTree so `az.summary(idata)` / `az.plot_*` work directly. Docs
explicitly warn it is unsuitable for skewed/multimodal posteriors.

**(2) `pymc_extras.find_MAP(..., compute_hessian=True)` — the explicit two-step.** Use when you want
the MAP and the inverse-Hessian covariance in hand:
```python
from pymc_extras.inference import find_MAP
with model:
    map_res = find_MAP(method="L-BFGS-B", compute_hessian=True,
                       gradient_backend="pytensor", random_seed=RANDOM_SEED)
# compute_hessian=True -> inverse Hessian at optimum included in the returned DataTree
# (needed for Laplace; can be expensive in high dim).
```
Then a MvNormal(mean=MAP, cov=inverse_Hessian) IS the Laplace approximation.

**(3) Stock `pm.find_MAP()`** (in core PyMC) returns only the **point estimate** dict — it does NOT
give you the Hessian/covariance, so on its own it is MAP, not Laplace. The bible's "pm.find_MAP +
Hessian" route historically meant pairing it with `model.compile_d2logp()` / a numeric Hessian and
inverting — that manual route still works but `pymc_extras.find_MAP(compute_hessian=True)` /
`fit_laplace` now supersede it. Recommend writers lead with `fit_laplace` and show the manual
Hessian only to demystify the mechanics.

> ⚠️ `pm.Laplace` exists in core PyMC but is a **distribution** (the double-exponential), totally
> unrelated to the Laplace *approximation*. Flag this name clash explicitly — it's a real footgun.

The Laplace covariance is on the **unconstrained (transformed) scale**. Positive params (sigma,
tau) are sampled internally on `log` scale; the Gaussian lives there and is back-transformed, which
is *why* QUAP can look decent for a positive scale param even though it can't be Gaussian on the
natural scale. McElreath's `quap` does the same trick.

### Variational inference in PyMC v5 — VERIFIED `pm.fit` API
```python
with model:
    approx = pm.fit(n=20000, method="advi", random_seed=RANDOM_SEED)  # mean-field ADVI (default)
    # method options confirmed: "advi", "fullrank_advi", "svgd", "asvgd"
idata = approx.sample(1000)                 # -> InferenceData; use with az.* directly
plt.plot(approx.hist)                        # negative-ELBO trajectory (the "loss") to eyeball convergence
approx.mean.eval(); approx.std.eval()        # variational params (mean-field)
```
Object-oriented form (equivalent, gives `.refine`):
```python
advi = pm.ADVI()                 # or pm.FullRankADVI(), pm.SVGD(n_particles=...), pm.ASVGD()
approx = advi.fit(20000)
advi.refine(50000)               # continue optimizing the SAME approximation
```
Convergence callback (VERIFIED import path):
```python
from pymc.variational.callbacks import CheckParametersConvergence
approx = pm.fit(method="advi",
                callbacks=[CheckParametersConvergence(diff="absolute")])  # also Tracker(...)
```
SVGD with custom optimizer (from the official quickstart):
```python
approx = pm.fit(300, method="svgd",
                inf_kwargs=dict(n_particles=1000),
                obj_optimizer=pm.sgd(learning_rate=0.01))
```
Minibatch ADVI (for big data — Ch 17 also touches this):
```python
X = pm.Minibatch(data, batch_size=500)
with pm.Model() as m:
    ...
    pm.Normal("lik", mu, sigma=sd, observed=X, total_size=data.shape[0])  # total_size IS REQUIRED
    approx = pm.fit()             # default 10_000 iters, mean-field ADVI
```
Posterior of a derived quantity / out-of-sample: `approx.sample_node(expr, size=..., more_replacements={Xt: X_test})`.

### Pathfinder — VERIFIED (`pymc_extras`)
Pathfinder is now PyTensor-native inside `pymc_extras`. Two equivalent entry points:
```python
import pymc_extras as pmx
with model:
    idata = pmx.fit(method="pathfinder", num_draws=1000, jitter=12, random_seed=RANDOM_SEED)
# OR the direct function:
from pymc_extras.inference import fit_pathfinder
with model:
    idata = fit_pathfinder(num_paths=4, num_draws=1000, importance_sampling="psis",
                           inference_backend="pymc", random_seed=RANDOM_SEED)
```
`pmx.fit(method="pathfinder", ...)` is a thin wrapper that calls `fit_pathfinder`. Returns an
InferenceData "just like `pm.sample()`", so `az.summary`, `az.plot_forest([idata_nuts, idata_path], ...)`
work. Key verified args: `num_paths=4`, `num_draws=1000`, `num_draws_per_path=1000`,
`importance_sampling ∈ {"psis"(default), "psir", "identity", None}`, `inference_backend ∈ {"pymc","blackjax"}`,
`jitter=2.0` (raise it + raise `num_paths` for tougher posteriors), `jacobian_correction=True`
(disabling it produces spuriously high Pareto-k). When `store_diagnostics`/`add_pathfinder_groups`
are set, it attaches a `pathfinder`/k̂ diagnostic group.

### NumPyro VI (for cross-reference / Ch 17 JAX path) — VERIFIED
```python
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import (AutoNormal, AutoDiagonalNormal,
                                      AutoMultivariateNormal, AutoLaplaceApproximation)
guide = AutoNormal(model)                     # mean-field
# guide = AutoMultivariateNormal(model)       # full-rank (captures correlations)
# guide = AutoLaplaceApproximation(model)     # MvNormal at MAP w/ -inverse-Hessian cov == Laplace
svi = SVI(model, guide, numpyro.optim.Adam(1e-3), Trace_ELBO())
svi_result = svi.run(rng_key, num_steps=5000)
```
`AutoLaplaceApproximation` is NumPyro's built-in Laplace; `AutoNormal`/`AutoDiagonalNormal` =
ADVI mean-field; `AutoMultivariateNormal` = full-rank ADVI analogue.

---

## Key concepts & equations the chapter must nail

**Grid → why it dies.** Posterior on a grid: `p(θ|y) ∝ p(y|θ)p(θ)` evaluated at each node, then
normalized by the (trapezoidal) sum. Cost ∝ `K^d` for `K` points in `d` dims. This single fact
motivates Laplace, MCMC, and VI.

**Laplace / quadratic approximation.** Taylor-expand the log-posterior `ℓ(θ)=log p(θ|y)` around the
mode `θ̂` (the MAP). First-order term vanishes (gradient = 0 at the mode), so:
$$\ell(\theta)\approx \ell(\hat\theta) - \tfrac12 (\theta-\hat\theta)^\top H (\theta-\hat\theta),\quad
H = -\nabla^2 \ell(\theta)\big|_{\hat\theta}.$$
Exponentiating, the posterior is approximated by a Gaussian:
$$p(\theta\mid y)\;\approx\;\mathcal N\!\big(\hat\theta,\; H^{-1}\big),$$
i.e. **mean = MAP, covariance = inverse of the negative Hessian of the log-posterior at the mode**.
This is *exactly* what McElreath's `quap` and PyMC's `fit_laplace` compute. ASCII:
`posterior ≈ Normal(MAP, inv(Hessian_of_neg_log_post))`.

When **valid**: posterior is unimodal and roughly Gaussian on the (transformed) scale — large-n
regimes (Bernstein–von Mises: posterior → Gaussian as data accumulates), well-identified GLMs with
plenty of data. When **invalid**: skewed/heavy-tailed posteriors, bounded params near a boundary,
**multimodality** (Laplace grabs one mode), low data / strong priors, hierarchical variance params
(the funnel — Laplace badly misses the neck), correlations the diagonal can't capture. The
inverse-Hessian also tends to **underestimate** tail width for skewed posteriors.

**KL divergence & the ELBO (the heart of Ch 06).** VI posits a family `q(θ;φ)` and minimizes the
**reverse** KL `KL(q‖p)` to the true posterior `p(θ|y)`. Start from the marginal likelihood:
$$\log p(y) = \underbrace{\mathbb E_q\!\left[\log\tfrac{p(\theta,y)}{q(\theta)}\right]}_{\text{ELBO}(q)} \;+\; \underbrace{\mathrm{KL}\!\big(q(\theta)\,\|\,p(\theta\mid y)\big)}_{\ge 0}.$$
Because `log p(y)` is a constant w.r.t. `q` and KL ≥ 0, **maximizing the ELBO = minimizing the KL**,
and the ELBO is a *lower bound* on the log-evidence. Equivalent decomposition:
$$\text{ELBO}(q)=\mathbb E_q[\log p(y\mid\theta)] - \mathrm{KL}(q(\theta)\,\|\,p(\theta)) = \mathbb E_q[\log p(\theta,y)] + \mathbb H[q].$$
PyMC reports `-ELBO` as `approx.hist` (the loss to minimize).

**Why reverse KL is mode-seeking / variance-underestimating.** `KL(q‖p)=∫ q log(q/p)`. Where
`q>0` but `p≈0` the integrand blows up, so `q` is *penalized for putting mass where p has none* →
`q` shrinks onto a single mode and **underestimates posterior variance / ignores correlations**
(mean-field). Forward KL `KL(p‖q)` is mass-covering instead (this is the EP/moment-matching flavor;
worth a one-line contrast). This is the #1 thing the chapter must hammer: **VI gives you the right
shape of the wrong (too narrow) distribution.**

**Mean-field vs full-rank ADVI (Kucukelbir et al. 2017).** ADVI = (i) transform constrained params
to unconstrained ℝ via bijectors, (ii) assume a Gaussian `q` in that space, (iii) maximize the ELBO
by stochastic gradient ascent using automatic differentiation + the reparameterization trick
(Monte-Carlo ELBO gradients). **Mean-field**: `q=∏ N(μ_i, σ_i²)` — diagonal covariance, no
correlations (fast, most biased). **Full-rank**: `q=N(μ, LLᵀ)` — dense covariance, captures
correlations (slower, O(d²) params, better but still underestimates variance vs MCMC).

**SVGD / ASVGD.** Stein Variational Gradient Descent: a particle method — a set of particles evolves
under a deterministic kernelized gradient flow that decreases KL; non-Gaussian, can capture
multimodality. `pm.SVGD(n_particles=...)`; **ASVGD** = Amortized SVGD (`method="asvgd"`). Mention as
"beyond-Gaussian VI" but note they're less battle-tested than ADVI/NUTS.

**Normalizing-flow VI (concept).** Replace the Gaussian `q` with a flexible distribution built by
pushing a simple base through a chain of invertible, differentiable maps (the change-of-variables /
log-det-Jacobian); strictly more expressive than ADVI. Treat conceptually — PyMC's historical
`NFVI`/`flows` API is legacy and writers should NOT assert a specific current spelling; point to
NumPyro `AutoBNAFNormal`/`AutoIAFNormal` if a concrete flow guide is wanted, and FLAG as
"check current docs."

**Pathfinder (Zhang, Carpenter, Gelman, Vehtari 2022).** Run L-BFGS from a random init; along the
optimization *path* it forms a cheap Gaussian (Laplace-like) approximation at each iterate using the
L-BFGS inverse-Hessian estimate; pick the iterate whose Gaussian has the **lowest estimated KL** to
the posterior; draw from it; **Pareto-smoothed importance resampling (PSIS/PSIR)** corrects the
draws. Multi-path Pathfinder runs several inits in parallel and PSIR-combines them to dodge bad
modes. It's much faster than NUTS, more robust than plain ADVI, and an excellent **NUTS initializer**.

---

## Common errors → causes → fixes

| Symptom | Cause | Fix |
|---|---|---|
| `pm.quap` / `pm.Laplace(approx)` not found | `quap` is a `rethinking` R concept; `pm.Laplace` is the *distribution* | Use `pymc_extras.fit_laplace` or `find_MAP(compute_hessian=True)`; never confuse with the distribution |
| `find_MAP` returns only a point, no covariance | stock `pm.find_MAP` gives MAP only | use `pymc_extras.find_MAP(compute_hessian=True)` or `fit_laplace`; or build Hessian via `model.compile_d2logp()` and invert |
| Laplace covariance not positive-definite / Hessian singular | mode at a boundary, flat ridge, non-identified params, or near a funnel neck | reparameterize (non-centered), add weakly-informative priors, standardize predictors, or abandon Laplace for NUTS |
| ADVI ELBO (`approx.hist`) noisy / not flattening | learning rate / too few iters / multimodal target | raise `n`, lower `obj_optimizer` lr, use `CheckParametersConvergence`, try `fullrank_advi`, or switch to NUTS |
| ADVI ELBO diverges to ±inf early | bad init / extreme priors / un-standardized data | standardize predictors, set sane priors, `pm.fit(..., obj_optimizer=pm.adagrad_window(...))` |
| VI posterior intervals far too narrow vs NUTS | reverse-KL variance underestimation (inherent) | expected — report it; use full-rank/Pathfinder; **validate with PSIS k̂**, fall back to NUTS for final inference |
| VI misses correlations / funnel | mean-field diagonal `q` | use `fullrank_advi`, non-centered reparameterization, or NUTS |
| Pathfinder "confident wrong answer" | single bad optimization path / strong multimodality / too-small jitter | raise `num_paths`, raise `jitter`, keep `importance_sampling="psis"`, check Pareto-k; verify against NUTS |
| Pathfinder high Pareto-k everywhere | `jacobian_correction=False` or genuinely poor Gaussian fit | keep `jacobian_correction=True`; if still high, the Gaussian family is inadequate → NUTS |
| Minibatch ADVI gives wrong posterior scale | missing `total_size` | always pass `total_size=N` on the observed node so the likelihood is rescaled |
| `az.loo` / model comparison fails after VI | no pointwise log-likelihood stored | sample with `idata_kwargs={"log_likelihood": True}` for the NUTS reference; for VI use the k̂-from-PSIS diagnostic (Yao 2018), not LOO |

---

## Datasets & load snippets (from the shared catalog, §7)

- **Globe tossing / Bernoulli** (synthetic) — ideal for **grid approximation** (1-D `p`). Just
  `successes, n = 6, 9`. Best first illustration; matches McElreath SR Ch 2.
- **Howell !Kung height** — Laplace/QUAP flagship (matches McElreath SR Ch 4, his canonical `quap`
  example):
  ```python
  import pandas as pd
  url = "https://raw.githubusercontent.com/rmcelreath/rethinking/master/data/Howell1.csv"
  d = pd.read_csv(url, sep=";")
  adults = d[d.age >= 18]   # height ~ Normal(mu, sigma); mu = a + b*(weight - weight.mean())
  ```
  Laplace is *excellent* here (lots of data, near-Gaussian posterior) — use it to show QUAP working,
  then show where it fails.
- **Eight Schools** (hardcoded) — the perfect place to show **VI/Laplace FAILING** on the funnel and
  Pathfinder doing better than ADVI:
  ```python
  import numpy as np
  y = np.array([28.,8.,-3.,7.,-1.,1.,18.,12.]); sigma = np.array([15.,10.,16.,11.,9.,11.,10.,18.])
  ```
  Use the **non-centered** parameterization (`z ~ Normal(0,1); theta = mu + tau*z`) and compare
  ADVI vs Pathfinder vs NUTS on `tau` — VI underestimates `tau` and the neck.
- **Logistic GLM (synthetic or Titanic/Penguins)** — clean case where mean-field ADVI and Laplace
  are close to NUTS (well-identified, large-n) → shows VI *succeeding*. Generate synthetic with known
  `beta` to show recovery.

> Teaching device the bible explicitly encourages: **synthetic data with a known generative
> process**, then show grid/Laplace/ADVI/Pathfinder all recovering (or failing to recover) the true
> params. Strongest pedagogy for an estimator chapter.

---

## Curated resources (real URLs, each with a why + pointer)

- **PyMC `fit_laplace` API** — https://www.pymc.io/projects/extras/en/stable/generated/pymc_extras.inference.fit_laplace.html
  — exact signature/defaults for the recommended Laplace call. Verify args here before writing code.
- **PyMC `find_MAP` API** — https://www.pymc.io/projects/extras/en/stable/generated/pymc_extras.inference.find_MAP.html
  — `compute_hessian=True` is the explicit Laplace ingredient; see the param note.
- **PyMC `fit_pathfinder` API** — https://www.pymc.io/projects/extras/en/stable/generated/pymc_extras.inference.fit_pathfinder.html
  — full Pathfinder signature (`num_paths`, `importance_sampling`, `inference_backend`).
- **Pathfinder example gallery** — https://www.pymc.io/projects/examples/en/latest/variational_inference/pathfinder.html
  — runnable `pmx.fit(method="pathfinder", ...)` on Eight Schools + `az.plot_forest` vs NUTS. Copy this idiom.
- **PyMC Variational API quickstart** — https://www.pymc.io/projects/examples/en/latest/variational_inference/variational_api_quickstart.html
  — the canonical `pm.fit`/`approx.sample`/`approx.hist`/`Tracker`/minibatch reference. Primary code source for Ch 06.
- **`pymc.fit` API** — https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.fit.html
  — confirms `method` strings: `advi`, `fullrank_advi`, `svgd`, `asvgd`.
- **`pymc.ADVI`** — https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.ADVI.html
  and **`pymc.FullRankADVI`** — https://www.pymc.io/projects/docs/en/latest/api/generated/pymc.FullRankADVI.html
  — OO interface, `.fit`, `.refine`, mean-field vs full-rank definitions.
- **Kucukelbir, Tran, Ranganath, Gelman, Blei (2017), "Automatic Differentiation Variational
  Inference," JMLR 18(14):1–45** — https://jmlr.org/papers/v18/16-107.html (arXiv:1603.00788) —
  the ADVI paper: transform→Gaussian→SGD ELBO. The §on mean-field vs full-rank is required reading.
- **Zhang, Carpenter, Gelman, Vehtari (2022), "Pathfinder: Parallel quasi-Newton variational
  inference," JMLR 23(306):1–49** — https://jmlr.org/papers/v23/21-0889.html (arXiv:2108.03782) —
  the Pathfinder paper; Fig/§ on KL-along-the-path and multi-path PSIR.
- **Yao, Vehtari, Simpson, Gelman (2018), "Yes, but Did It Work?: Evaluating Variational Inference,"
  ICML** — https://arxiv.org/abs/1802.02538 (PMLR: http://proceedings.mlr.press/v80/yao18a/yao18a.pdf)
  — THE diagnostic paper: PSIS k̂ for VI (k̂ > 0.7 ⇒ distrust VI) and VSBC. Use its k̂ thresholds.
- **Stan Reference Manual — Pathfinder** — https://mc-stan.org/docs/reference-manual/pathfinder.html
  — Stan's authoritative description; good for the CmdStanPy cross-reference the bible wants.
- **NumPyro AutoGuide docs** — https://num.pyro.ai/en/stable/autoguide.html
  — `AutoNormal`/`AutoMultivariateNormal`/`AutoLaplaceApproximation`/`AutoBNAFNormal`; JAX VI cross-ref.
- **McElreath `quap` reference** — https://rdrr.io/github/rmcelreath/rethinking/man/quap.html and
  source https://github.com/rmcelreath/rethinking/blob/master/R/map-quap.r — confirms `quap` =
  mode (MAP) + quadratic curvature (Hessian); renamed from `map` in SR 2nd ed. Cite SR Ch 2 & 4.
- **PyMC Discourse: "Laplace approximation vs VI"** — https://discourse.pymc.io/t/laplace-approximation-vs-vi/17722
  — practitioner guidance on when to reach for each.
- **ArviZ summary / diagnostics** — https://python.arviz.org/en/stable/api/generated/arviz.summary.html
  — `az.summary` works on VI/Laplace/Pathfinder InferenceData; Pareto-k bands (≤0.5 good, ≤0.7 ok, >0.7 bad).

Supplementary (from the bible's canonical list, good for theory framing):
- **Blei, Kucukelbir, McAuliffe (2017), "Variational Inference: A Review for Statisticians"** —
  https://arxiv.org/abs/1601.00670 — the canonical VI review; ELBO/CAVI derivation, mean-field, the
  variance-underestimation discussion. Best single citation for the ELBO/KL section.
- **Bayesian Modeling & Computation in Python (Martin/Kumar/Lao)** — https://bayesiancomputationbook.com
  — Ch 11 (the "what's-under-the-hood" inference chapter) covers Laplace, VI, grid; idiomatic PyMC/ArviZ.

---

## Uncertainties / things the writer should double-check

- **`pymc_extras` is fast-moving.** Re-verify `fit_laplace` / `find_MAP` / `fit_pathfinder` signatures
  against the *installed* version at write time (docs showed 0.9.x–0.10.x). The high-level
  `pmx.fit(method=...)` wrapper and the direct `fit_*` functions both exist; confirm both still
  dispatch as documented. Package was formerly `pymc-experimental`; `import pymc_extras as pmx` is current.
- **Laplace return container.** Docs say `fit_laplace`/`find_MAP` return a **`DataTree`** (arviz
  DataTree). Treat it like InferenceData for `az.*`, but if a writer hits an attribute error,
  check whether `.to_inferencedata()` / group access differs by version. Don't over-assert internal group names.
- **Normalizing-flow VI in PyMC.** The legacy `pm.NFVI` / flow-formula API is essentially deprecated
  /unmaintained. Do NOT present a concrete current PyMC flow-VI call as verified. Cover NF
  *conceptually* and point concrete flow guides to NumPyro (`AutoIAFNormal`/`AutoBNAFNormal`),
  flagging "check current docs."
- **ASVGD.** `method="asvgd"` exists historically; confirm it's still wired in the installed PyMC
  before showing runnable code (SVGD is the safer, better-documented particle method to demo).
- **k̂-for-VI in PyMC.** Yao 2018 defines the PSIS k̂ VI diagnostic; Pathfinder stores Pareto-k via
  its importance-sampling step, but PyMC's *generic* ADVI output does not auto-attach a VI k̂. To
  compute it for plain ADVI you must run PSIS on importance weights `log p(θ,y) − log q(θ)` manually
  (or just compare to NUTS). Verify whether a current ArviZ/pymc_extras helper exposes this before
  claiming a one-liner.
- **`np.trapz` deprecation.** Newer NumPy renames to `np.trapezoid`; use whichever matches the pinned NumPy.
- **MAP ≠ posterior mean** under transforms — be careful: `find_MAP` optimizes on the unconstrained
  scale, so the reported MAP for a positive param is the mode there, back-transformed. State this so
  learners don't compare it naively to the natural-scale posterior mean.
