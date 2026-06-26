# R10 — Mixture Models + Nonparametric / Flexible Models

Research dossier for **Chapter 13 (Mixture models & latent variables)** and **Chapter 14
(Nonparametric & flexible models)**. Reference material for chapter writers — verify nothing
here against memory; it is checked against current docs (June 2026). Versions assumed: PyMC v5,
ArviZ, Bambi, CmdStanPy, pymc-bart. Where I could not verify a symbol I say so explicitly.

---

## Scope & which chapters use this

- **Ch 13** consumes: finite Gaussian mixtures (`pm.Mixture`, `pm.NormalMixture`); Dirichlet prior
  on weights; **identifiability & label switching** + the **ordered-transform fix**; marginalizing
  discrete latent assignments (PyMC marginalizes automatically; the **Stan `log_sum_exp` / `target +=`**
  idiom); zero-inflated & hurdle models (`pm.ZeroInflatedPoisson`, `pm.HurdlePoisson`); latent-variable
  thinking. Datasets: **Old Faithful / geyser** (density estimation), synthetic 3-cluster Gaussian,
  a count dataset with excess zeros (e.g. fishing/manuscripts-style synthetic à la McElreath).
- **Ch 14** consumes: **Bayesian splines** (patsy `dmatrix` B-splines + Normal or Gaussian-random-walk
  coefficients; cherry-blossom/`bikes`/births examples); **BART** via `pymc_bart` (`pmb.BART`, PGBART
  sampling, variable importance, partial dependence); **Dirichlet-process / infinite mixtures** via
  truncated stick-breaking; and the judgment call **splines vs GP vs BART**.
- The two chapters share the "flexible function approximation" thread; Ch 14 should explicitly bridge
  back to **Ch 12 (Gaussian processes)** — splines, GPs, and BART are three answers to the same
  "I don't want to commit to a parametric mean function" question.

---

## Verified API & idioms (confirmed against current docs)

### 1. Finite Gaussian mixture with the ordered transform (the canonical Ch 13 model)

The **current PyMC example-gallery spelling** (Gaussian Mixture Model page) uses
`pm.distributions.transforms.univariate_ordered`:

```python
import pymc as pm
import numpy as np

k = 3
with pm.Model(coords={"cluster": range(k)}) as model:
    μ = pm.Normal(
        "μ", mu=0, sigma=5,
        transform=pm.distributions.transforms.univariate_ordered,
        initval=[-4, 0, 4],          # MUST be sorted; required or sampler errors
        dims="cluster",
    )
    σ = pm.HalfNormal("σ", sigma=1, dims="cluster")
    weights = pm.Dirichlet("w", np.ones(k), dims="cluster")
    pm.NormalMixture("x", w=weights, mu=μ, sigma=σ, observed=x)
    idata = pm.sample()
```

> ⚠️ **VERIFY THE TRANSFORM NAME.** In PyMC's `transforms.py`, `univariate_ordered` and
> `multivariate_ordered` are now **deprecated aliases** that emit a `FutureWarning` and forward to a
> single `ordered` transform (handled via a module `__getattr__`). So **both** `univariate_ordered`
> (gallery spelling, still works) and the newer `pm.distributions.transforms.ordered` exist. The class
> is `pm.distributions.transforms.Ordered` (it has `ascending`/`positive` options). Writer guidance:
> show `univariate_ordered` (matches the live gallery) but add a `# newer alias: transforms.ordered`
> comment, and note the deprecation so code keeps running across point releases.

Key invariants the chapter must state:
- `initval` for the ordered means **must be strictly increasing**, else you get an initialization /
  "starting point ... not finite" error.
- Dirichlet concentration `np.ones(k)` is the symmetric/uniform prior on the simplex; larger values
  push weights toward equality, smaller (<1) toward sparse/near-degenerate weights.
- `pm.NormalMixture` is the convenience wrapper; `mu`, `sigma`, `tau` are per-component and broadcast
  against `w` along the last axis.

### 2. General `pm.Mixture` with `comp_dists` (verified signature)

`comp_dists` components **must be created with the `.dist()` API** (unregistered distributions), not as
named model variables:

```python
with pm.Model() as model:
    w = pm.Dirichlet("w", a=np.array([1, 1]))
    lam1 = pm.Exponential("lam1", lam=1)
    lam2 = pm.Exponential("lam2", lam=1)
    components = [pm.Poisson.dist(mu=lam1), pm.Poisson.dist(mu=lam2)]  # list of dists
    like = pm.Mixture("like", w=w, comp_dists=components, observed=data)
```

Two valid `comp_dists` shapes (state both — a common confusion):
- **List form** — heterogeneous families: `[pm.Normal.dist(mu=m, sigma=1), pm.StudentT.dist(nu=4, mu=m, sigma=1)]`.
- **Single batched dist** — homogeneous, vectorized over components on the **last** axis:
  `pm.Normal.dist(mu=μ, sigma=σ, shape=(k,))` or `pm.Poisson.dist(mu=pm.math.stack([lam1, lam2]), shape=(2,))`.
  When batched, `w` aligns with that last axis.

`pm.Mixture` **marginalizes the discrete component label automatically** — there is no explicit
categorical latent `z`. This is the headline contrast with Stan. (PyMC also has experimental explicit
marginalization via `pm.MarginalModel` / the `marginalize` machinery for models where you *do* declare
discrete latents — see the "Automatic marginalization of discrete variables" how-to — but for mixtures
`pm.Mixture` already does it.)

### 3. Zero-inflated & hurdle counts (verified)

```python
y = pm.ZeroInflatedPoisson("y", psi=psi, mu=mu, observed=counts)
# psi = P(structural-process active); (1-psi) extra zeros. mu = Poisson rate.
y = pm.ZeroInflatedNegativeBinomial("y", psi=psi, mu=mu, alpha=alpha, observed=counts)
y = pm.ZeroInflatedBinomial("y", psi=psi, n=n, p=p, observed=counts)
```

Hurdle distributions exist as first-class PyMC v5 distributions:
```python
y = pm.HurdlePoisson("y", psi=psi, mu=mu, observed=counts)
y = pm.HurdleNegativeBinomial("y", psi=psi, mu=mu, alpha=alpha, observed=counts)
y = pm.HurdleGamma(...)  # continuous-with-point-mass-at-0 relative
```

> 🧮 **ZIP vs Hurdle — the chapter MUST nail the distinction.**
> - **Zero-inflated**: zeros come from *two* sources — a structural "always-zero" process **and** the
>   count process (which can also emit 0). PMF: `P(0) = (1-ψ) + ψ·Poisson(0|μ)`; `P(y>0) = ψ·Poisson(y|μ)`.
> - **Hurdle**: a clean two-part split — one Bernoulli gate decides zero vs positive, and positives are
>   drawn from a **zero-truncated** count distribution. The count process can **never** emit 0.
>   PMF: `P(0)=1-ψ`; `P(y) = ψ·Poisson(y|μ)/(1−Poisson(0|μ))` for y≥1.
>   (Note PyMC's `psi` convention: `psi` is the prob of the *non-zero / above-hurdle* branch.)

### 4. Stan finite-mixture idiom (marginalize the label — verified against Stan User's Guide)

```stan
parameters {
  simplex[K] theta;            // mixing proportions
  ordered[K] mu;               // ordered locations -> identifiability
  vector<lower=0>[K] sigma;    // scales
}
model {
  vector[K] log_theta = log(theta);
  for (n in 1:N) {
    vector[K] lps = log_theta;
    for (k in 1:K)
      lps[k] += normal_lpdf(y[n] | mu[k], sigma[k]);
    target += log_sum_exp(lps);
  }
}
```

Teaching points: `ordered[K] mu` is Stan's *native* fix for label switching (declared type, no separate
transform). `log_sum_exp` is the numerically stable `log Σ exp`. For exactly **two** components the
sugar `target += log_mix(lambda, lpdf1, lpdf2)` exists. **Stan cannot sample discrete parameters**, so
marginalization is mandatory, not an optimization — contrast with PyMC where `pm.Mixture` does the same
summation for you under the hood.

### 5. Bayesian splines (Ch 14) — verified gallery pattern

```python
import numpy as np
from patsy import dmatrix

num_knots = 15
knot_list = np.quantile(blossom_data.year, np.linspace(0, 1, num_knots))
B = dmatrix(
    "bs(year, knots=knots, degree=3, include_intercept=True) - 1",
    {"year": blossom_data.year.values, "knots": knot_list[1:-1]},
)

with pm.Model(coords={"splines": np.arange(B.shape[1])}) as spline_model:
    a = pm.Normal("a", 100, 5)
    w = pm.Normal("w", mu=0, sigma=3, size=B.shape[1], dims="splines")
    mu = pm.Deterministic("mu", a + pm.math.dot(np.asarray(B, order="F"), w.T))
    sigma = pm.Exponential("sigma", 1)
    D = pm.Normal("D", mu=mu, sigma=sigma, observed=blossom_data.doy)
    idata = pm.sample()
```

- `np.asarray(B, order="F")` (Fortran/column-major) avoids a slow/copy path in the matmul — the gallery
  does this; mention it.
- **Two prior choices for `w`** the chapter should contrast:
  1. **Independent Normal** (gallery default): `pm.Normal("w", 0, 3, dims="splines")` — simple, but
     adjacent coefficients are unconstrained → can be wiggly.
  2. **Gaussian random walk (P-spline-flavored smoothing prior)**:
     `w = pm.GaussianRandomWalk("w", sigma=tau, shape=B.shape[1])` with a HalfCauchy/Exponential `tau`.
     This ties adjacent knots together (penalizes 1st/2nd differences) → smoother fits; the smoothing
     strength is learned. This is the "Normal random-walk coeffs" the topic brief asks for. The BMCP
     Ch 5 splines treatment and the births/bikes examples use this smoothing-prior idea.

> ⚠️ Don't put `dims=` AND `size=` that disagree. The gallery uses `size=B.shape[1], dims="splines"`
> with a matching coord. Keep `B.shape[1]` and the `splines` coord length identical.

### 6. BART via `pymc_bart` (VERIFIED current API)

```python
import pymc as pm
import pymc_bart as pmb
import numpy as np

with pm.Model() as model_bikes:
    α = pm.Exponential("α", 1)
    μ = pmb.BART("μ", X, np.log(Y), m=50)               # m = number of trees
    y = pm.NegativeBinomial("y", mu=pm.math.exp(μ), alpha=α, observed=Y)
    idata = pm.sample(tune=500, draws=2000,
                      compute_convergence_checks=False, random_seed=RANDOM_SEED)
```

**Verified `pmb.BART` constructor:** `pmb.BART(name, X, Y, m=50, alpha=0.95, beta=2.0,
response='constant', split_rules=None, split_prior=None)`.
- `X`: covariate matrix (n×p), `Y`: response vector; `m`: #trees (more trees = smoother, slower).
- `alpha`, `beta`: control the tree-depth prior `P(node splits at depth d) = alpha (1+d)^(-beta)` —
  the regularization that keeps each tree weak.
- `response`: leaf-value method (`'constant'` default; `'linear'` exists).
- `split_rules`, `split_prior`: per-column split-rule control / covariate-selection weights.
- BART output is a **latent function on the link scale** — wrap it (`pm.math.exp`, `pm.math.invlogit`)
  before the likelihood. For Poisson/NegBin counts pass `Y=np.log(Y)`; for Bernoulli use `invlogit(μ)`.
- `pm.sample()` **auto-detects the BART RV and assigns the PGBART (Particle Gibbs) step**; you do not
  pass a sampler. `compute_convergence_checks=False` is commonly set because per-tree params aren't the
  inferential target. NUTS handles the other continuous params alongside PGBART.

**Verified plotting / utility functions (`pymc_bart` namespace):**
- `pmb.plot_pdp(bartrv, X, Y=None, xs_interval='quantiles', var_idx=..., var_discrete=..., func=None,
  samples=200, grid='long', ...)` — partial dependence plots. `func=np.exp` to map back from log scale.
- `pmb.plot_ice(...)` — individual conditional expectation curves (same call shape as `plot_pdp`).
- `pmb.compute_variable_importance(idata, bartrv, X, method='VI', samples=50, random_seed=None)` →
  returns a results object; then `pmb.plot_variable_importance(vi_results, ...)`.
  The `method` includes a **backward-search** variant (`method='backward'`) that recomputes R² dropping
  one variable at a time; the default counting heuristic counts inclusion frequency across trees.
- `pmb.plot_variable_inclusion(idata, X, labels=None, ...)` and `pmb.get_variable_inclusion(...)`.
- `pmb.plot_scatter_submodels(vi_results, ...)`.

```python
pmb.plot_pdp(μ, X=X, Y=Y, grid=(2, 2), func=np.exp, var_discrete=[3])
vi_results = pmb.compute_variable_importance(idata, μ, X)
pmb.plot_variable_importance(vi_results)
```

### 7. Dirichlet-process / infinite mixture via truncated stick-breaking (VERIFIED, PyTensor)

```python
import pytensor.tensor as pt

def stick_breaking(beta):
    portion_remaining = pt.concatenate([[1], pt.extra_ops.cumprod(1 - beta)[:-1]])
    return beta * portion_remaining

K = 30  # truncation level
with pm.Model(coords={"component": range(K), "obs_id": range(N)}) as model:
    alpha = pm.Gamma("alpha", 1.0, 1.0)                    # DP concentration
    beta  = pm.Beta("beta", 1.0, alpha, dims="component")  # stick fractions
    w     = pm.Deterministic("w", stick_breaking(beta), dims="component")
    # component params, e.g. Normal-Gamma:
    tau   = pm.Gamma("tau", 1.0, 1.0, dims="component")
    lambda_ = pm.Gamma("lambda_", 10.0, 1.0, dims="component")
    mu    = pm.Normal("mu", 0, tau=..., dims="component")
    obs = pm.NormalMixture("obs", w, mu, tau=lambda_ * tau,
                           observed=data, dims="obs_id")
```

- Uses **`pytensor.tensor as pt`** (NOT aesara, NOT theano) — `pt.extra_ops.cumprod`, `pt.concatenate`.
  This is the #1 thing to get right when porting old PyMC3 DP-mixture notebooks.
- Truncation `K` is a modeling choice; check that the largest-index components carry negligible weight
  (`az.plot_forest` on `w`) — if the last sticks have real mass, raise `K`.
- DP-mixture posteriors are notoriously **multimodal / label-switchy**; expect divergences and weak R̂.
  The gallery accepts this for density estimation (the *density* is identified even if labels aren't).

---

## Key concepts & equations the chapter must nail

- **Finite mixture density:** `p(y) = Σ_{k=1}^K w_k · f(y | θ_k)`, `w` on the simplex (`Σ w_k = 1`,
  `w_k ≥ 0`). Latent label `z_n ~ Categorical(w)`, `y_n | z_n=k ~ f(·|θ_k)`.
- **Marginalized likelihood (what PyMC/Stan actually fit):**
  `p(y_n | w, θ) = Σ_k w_k f(y_n | θ_k)`, log form `log Σ_k exp(log w_k + log f(y_n|θ_k))` =
  `log_sum_exp`. Marginalizing the discrete `z` is what lets HMC/NUTS run (continuous geometry) and
  gives better mixing + tail exploration than Gibbs-sampling `z`.
- **Responsibilities (post-hoc, if you want cluster assignments back):**
  `r_{nk} = w_k f(y_n|θ_k) / Σ_j w_j f(y_n|θ_j)` — compute from posterior draws after fitting.
- **Non-identifiability / label switching:** the likelihood is invariant under permuting component
  labels → `K!` equivalent modes. Symptoms: bimodal/multimodal trace plots for `μ`, R̂ ≫ 1.01 on `μ`/`w`,
  forest plots where components overlap. **Fix:** break the symmetry by ordering — `ordered` transform
  on `μ` (PyMC) or `ordered[K] mu` (Stan). Caveat (from PyMC Discourse): the ordered transform can
  occasionally *worsen* geometry or add spurious peaks if components genuinely overlap; an alternative
  is post-hoc relabeling, but the ordered constraint is the default to teach.
- **Stick-breaking (GEM) construction of DP weights:** `β_k ~ Beta(1, α)`,
  `w_k = β_k ∏_{j<k}(1 − β_j)`. Concentration `α`: small → few effective clusters, large → many.
  Expected #occupied components ≈ `α log N`.
- **BART:** `f(x) ≈ Σ_{j=1}^m g_j(x; T_j, M_j)` — sum of `m` shallow regression trees, each a *weak*
  learner via the depth-penalizing prior (`α(1+d)^(-β)`, default α=0.95, β=2). Inference by Particle
  Gibbs (PGBART). Posterior is over the *function*, giving credible bands for free.
- **B-spline regression:** design matrix `B` (n×(#basis)) from cubic B-splines on chosen knots;
  `μ = a + B w`. Smoothing prior on `w` (random walk / 2nd-difference penalty) ↔ P-splines.
- **Splines vs GP vs BART (the Ch 14 judgment table):**
  - *Splines*: cheap, interpretable, great for 1–2D smooth trends; you choose knots/basis; extrapolation
    is poor; smoothness controlled by the coefficient prior.
  - *GP*: principled smoothness via a kernel, exact uncertainty, elegant in low-D; O(n³) cost
    (use HSGP/sparse — cross-ref Ch 12); kernel choice matters; struggles in high-D.
  - *BART*: handles many predictors, interactions, and nonlinearities automatically with little tuning;
    discontinuous/step-like fits; weaker on smooth extrapolation; gives variable importance + PDPs.
  Rule of thumb to state: smooth low-D trend → spline/GP; tabular many-feature predictive model with
  interactions → BART; need exact-ish uncertainty + smoothness in low-D → GP.

---

## Common errors -> causes -> fixes

| Symptom | Cause | Fix |
|---|---|---|
| `SamplingError: Initial evaluation ... not finite` on a mixture | `initval` for ordered means not strictly increasing, or σ init invalid | Pass sorted `initval=[-4,0,4]`; ensure `HalfNormal`/positive scale; check Dirichlet `a>0` |
| Bimodal traces, R̂≈2 on `μ`/`w`, components "swap" between chains | Label switching (permutation non-identifiability) | Add `transform=pm.distributions.transforms.univariate_ordered` (or `ordered`) on component means; in Stan declare `ordered[K] mu` |
| Ordered transform makes convergence *worse* / new peaks | Components genuinely overlap; ordering forces a poor coordinate system | Reduce K, use stronger priors separating means, or accept and relabel post-hoc; see Discourse thread |
| `comp_dists` raises about registered variables | Built components with `pm.Normal("c", ...)` (named RVs) instead of `.dist()` | Use the **`.dist()` API**: `pm.Normal.dist(mu=..., sigma=...)` |
| `w` and components misalign / shape error in `pm.Mixture` | Mixture axis must be the **last** axis; `w` length ≠ component axis | Build batched dist with `shape=(k,)` and matching Dirichlet `a=np.ones(k)` |
| Many divergences in DP mixture; weak ESS | Inherent multimodality + funnel between `α`, sticks, components | Raise `target_accept` (0.95–0.99); reparameterize; accept that the *density* is identified even if labels aren't; raise/lower K |
| `cumprod`/`concatenate` import errors porting a DP notebook | Old code uses `theano.tensor`/`aesara` | Use `import pytensor.tensor as pt`; `pt.extra_ops.cumprod`, `pt.concatenate` |
| Last stick-breaking components carry real weight | Truncation `K` too small | Increase `K`; verify tail weights ≈ 0 via `az.plot_forest(idata, var_names=["w"])` |
| Wiggly / overfit spline | Independent `Normal` coeff prior too loose; too many knots | Tighten `sigma`; use a `GaussianRandomWalk` smoothing prior on `w`; fewer knots |
| Spline matmul slow / shape mismatch | `B` not column-major; `dims` length ≠ `B.shape[1]` | `np.asarray(B, order="F")`; keep coord length == `B.shape[1]` |
| BART: divergence/convergence warnings spam | Per-tree params aren't smooth NUTS targets | `pm.sample(..., compute_convergence_checks=False)`; rely on PPCs not R̂ for the trees |
| BART fit on wrong scale (negatives in a count model) | Forgot the link wrap | Pass `Y=np.log(Y)` and wrap `mu=pm.math.exp(μ)` (Poisson/NegBin) or `invlogit` (Bernoulli) |
| ZIP fits look like plain Poisson | Confusing ZIP (zeros from 2 sources) with Hurdle (clean gate + truncated count) | Pick the right family; remember PyMC `psi` = prob of the active/above-hurdle branch |

---

## Datasets & load snippets

```python
# Old Faithful / geyser — mixtures & 1-D density estimation (Ch 13, DP mixture)
import seaborn as sns
geyser = sns.load_dataset("geyser")          # columns: duration, waiting, kind
x = geyser["waiting"].to_numpy().astype(float)
# (BMCP/PyMC DP example often standardizes: x = (x - x.mean())/x.std())

# Synthetic 3-cluster Gaussian (recovery demo — ENCOURAGED, known truth)
import numpy as np
rng = np.random.default_rng(8927)
true_w, true_mu, true_sd = [0.35, 0.40, 0.25], [-4., 0., 5.], [1.0, 1.2, 0.8]
z = rng.choice(3, size=600, p=true_w)
x = rng.normal(np.array(true_mu)[z], np.array(true_sd)[z])

# Excess-zero counts (zero-inflated / hurdle). Synthetic, McElreath-style:
psi_true, lam_true = 0.30, 3.0           # 30% structural zeros
active = rng.random(500) < psi_true
counts = np.where(active, rng.poisson(lam_true, 500), 0)

# Bikes (splines & BART count GLM). PyMC ships it:
import pymc as pd_get
bikes = pd.read_csv(pm.get_data("bikes.csv"))   # hour, temperature, humidity, weekday, count
# (Bambi also exposes a bikes-style dataset; the pymc-bart bikes example uses these features.)

# Cherry-blossom (classic Bayesian-spline example, McElreath/PyMC gallery):
# pm.get_data("cherry_blossoms.csv")  -> columns include year, doy (day-of-year of first bloom)
blossom = pd.read_csv(pm.get_data("cherry_blossoms.csv")).dropna(subset=["doy"])

# Coal mining disasters (change-point / Poisson, bridges Ch 13<->15):
coal = pd.read_csv(pm.get_data("coal.csv"))
```

> Verify `pm.get_data("bikes.csv")`, `"cherry_blossoms.csv")`, `"coal.csv")` resolve in the installed
> PyMC version (these are standard but the bundled set varies). Seaborn `geyser` is stable. Synthetic
> data with known truth is preferred for the identifiability/label-switching demo since you can show
> recovery of `true_mu` after the ordered constraint.

---

## Curated resources (each with a real URL + why + pointer)

- **PyMC — Gaussian Mixture Model (gallery):** https://www.pymc.io/projects/examples/en/latest/mixture_models/gaussian_mixture_model.html — canonical `pm.NormalMixture` + `univariate_ordered` + Dirichlet; copy the model skeleton for Ch 13.
- **PyMC — Marginalized Gaussian Mixture Model:** https://www.pymc.io/projects/examples/en/latest/mixture_models/marginalized_gaussian_mixture_model.html — the "why marginalize the label" narrative (slow mixing/tail exploration); note this page's code is older PyMC3-flavored, modernize it.
- **PyMC — Automatic marginalization of discrete variables (how-to):** https://www.pymc.io/projects/examples/en/latest/howto/marginalizing-models.html — for explicit discrete-latent marginalization beyond `pm.Mixture`.
- **PyMC — Dirichlet process mixtures for density estimation:** https://www.pymc.io/projects/examples/en/latest/mixture_models/dp_mix.html — verified stick-breaking with `pt.extra_ops.cumprod`; truncation, `alpha~Gamma`. Direct source for Ch 14 DP section.
- **PyMC — Dependent density regression:** https://www.pymc.io/projects/examples/en/latest/mixture_models/dependent_density_regression.html — DP mixtures as flexible regression; optional Ch 14 extension.
- **PyMC API — `pm.Mixture`:** https://www.pymc.io/projects/docs/en/latest/api/distributions/generated/pymc.Mixture.html — verified `w`/`comp_dists` + `.dist()` requirement and list-vs-batched forms.
- **PyMC API — `pm.NormalMixture`:** https://www.pymc.io/projects/docs/en/latest/api/distributions/generated/pymc.NormalMixture.html — per-component `mu`/`sigma`/`tau` semantics.
- **PyMC API — `pm.HurdlePoisson`:** https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.HurdlePoisson.html — exact PMF and the `psi` convention; pair with `ZeroInflatedPoisson` page for the ZIP-vs-hurdle contrast.
- **PyMC API — `transforms.Ordered` + `transforms.py` source:** https://www.pymc.io/projects/docs/en/latest/api/distributions/generated/pymc.distributions.transforms.Ordered.html and https://github.com/pymc-devs/pymc/blob/main/pymc/distributions/transforms.py — confirms `univariate_ordered`→`ordered` deprecation; cite for the "which spelling?" footnote.
- **PyMC — Splines (how-to):** https://www.pymc.io/projects/examples/en/latest/howto/spline.html — verified patsy `dmatrix bs(...)`, `np.asarray(B, order="F")`, `pm.math.dot`. The Ch 14 spline backbone.
- **PyMC-BART — API reference:** https://www.pymc.io/projects/bart/en/latest/api_reference.html — verified `pmb.BART(...)`, `plot_pdp`, `plot_ice`, `compute_variable_importance`, `plot_variable_importance`.
- **PyMC-BART — Introduction example:** https://www.pymc.io/projects/bart/en/latest/examples/bart_introduction.html — verified coal (Poisson) + bikes (NegBin) BART models, PDP & variable-importance calls.
- **PyMC-BART — GitHub repo:** https://github.com/pymc-devs/pymc-bart — install, version, `split_rules` details for the appendix.
- **Quiroga, Garay, Alonso, Loyola, Martin — "Bayesian additive regression trees for probabilistic programming" (2022/2023):** https://arxiv.org/abs/2206.03619 — THE pymc-bart paper; cite for BART-as-PPL-primitive, PGBART, the depth prior. (📜 Citation/Origin callout.)
- **Stan User's Guide — Finite Mixtures:** https://mc-stan.org/docs/stan-users-guide/finite-mixtures.html — verified `ordered[K] mu`, `simplex[K] theta`, `log_sum_exp` / `log_mix` idiom for the Stan box.
- **Stan User's Guide — Latent Discrete Parameters:** https://mc-stan.org/docs/stan-users-guide/latent-discrete.html — why discrete params must be summed out; broader marginalization context.
- **Bayesian Modeling and Computation in Python — Ch 5 Splines:** https://bayesiancomputationbook.com/markdown/chp_05.html — splines + GRW smoothing prior; free, the course's PyMC companion.
- **Bayesian Modeling and Computation in Python — Ch 7 BART:** https://bayesiancomputationbook.com/markdown/chp_07.html — BART chapter by the pymc-bart authors; intuition + diagnostics.
- **Bayesian Analysis with Python (3rd ed) — Ch on Mixture Models:** https://github.com/aloctavodia/BAP — Martin's mixtures chapter (finite mixtures, non-identifiability, ZIP, DP); the BMCP book has NO standalone mixtures chapter, so this is the book-of-record for Ch 13.
- **McElreath, Statistical Rethinking 2e — Ch 12 "Monsters and Mixtures":** https://github.com/rmcelreath/rethinking — zero-inflated Poisson (monks), over-dispersion, ordered categories; Ch 4 introduces smoothing splines (cherry blossoms). Cite both.
- **PyMC Discourse — ordered transform & label switching:** https://discourse.pymc.io/t/why-does-transform-pm-distributions-transforms-ordered-lead-to-worse-convergence/7944 and https://discourse.pymc.io/t/issues-with-multivariate-gmm-model-and-label-switching/14290 — real failure modes + fixes for the Common-Errors box.

---

## Uncertainties / things the writer should double-check

1. **Ordered transform spelling.** Live gallery uses `pm.distributions.transforms.univariate_ordered`
   (works, may emit `FutureWarning`); the consolidated name is `pm.distributions.transforms.ordered`
   (class `Ordered`). Confirm in the *installed* version which is non-deprecated and show that one;
   keep the other as a comment. I verified the deprecation exists in `transforms.py` but the exact
   release where `univariate_ordered` is fully removed is unconfirmed.
2. **`pm.get_data` bundled CSV names** (`bikes.csv`, `cherry_blossoms.csv`, `coal.csv`). Standard but
   the bundled set has drifted across versions — run them before relying on them; fall back to seaborn
   `geyser` and synthetic data, which are guaranteed.
3. **pymc-bart `response` options and `method=` values.** Verified `'constant'` default and a backward
   variable-importance search; the full enumerated set of `response` values and `compute_variable_importance`
   `method` strings should be re-read from the API page at writing time (active development repo).
4. **BMCP has no dedicated mixtures chapter** — splines = Ch 5, BART = Ch 7, and Ch 6 is *Time Series*.
   For Ch 13 mixtures, cite **Bayesian Analysis with Python (Martin)** instead. Don't cite a BMCP
   "mixtures chapter" — it doesn't exist.
5. **Bikes dataset provenance** differs between Bambi's example data and the pymc-bart bikes example
   (feature columns may differ). Pin to one loader and list its columns explicitly in the chapter.
6. **`pm.Mixture` batched-component axis** is the last axis — verify broadcasting in the exact model you
   ship (especially with `dims`), since a silent misalignment fits a wrong model without erroring.
7. I did **not** independently run any of this code; all snippets are transcribed from official docs/
   gallery/API pages. Treat them as verified-against-docs, not verified-by-execution.
