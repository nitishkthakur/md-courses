# R9 — Gaussian Processes (Research Dossier)

> Feeds **Chapter 12 — Gaussian Processes** (`12-gaussian-processes.md`). Also touches Ch 14
> (nonparametric/splines/BART connection), Ch 15 (structural time series / Mauna Loa style
> decompositions), and Ch 17 (HSGP/sparse for scaling). Verified against PyMC **v5.25–5.27**
> docs (stable, June 2026). Where I could not verify a symbol I say so explicitly.

---

## Scope & which chapters use this

- **Ch 12 (primary)**: GP intuition (distribution over functions), mean & covariance/kernel
  functions, GP regression with Gaussian likelihood (`pm.gp.Marginal`), latent GPs for
  non-Gaussian likelihoods (`pm.gp.Latent`), priors on lengthscale/amplitude/noise, kernel
  combination (Mauna Loa), the O(n³) scaling problem, sparse (`MarginalApprox`) and HSGP
  approximations, GP↔spline↔BART connections.
- **Ch 14**: GP↔spline equivalence (Kimeldorf–Wahba), GP↔BART contrast, "when to reach for
  which flexible model."
- **Ch 15**: additive-kernel decomposition of time series (trend + seasonal + noise) is the GP
  view of structural time series; HSGP is the practical engine for long series.
- **Ch 17**: HSGP and `MarginalApprox` as the canonical scaling recipes; `nuts_sampler=` backends.

---

## Verified API & idioms (confirmed against current PyMC docs)

**GP implementation classes** (in `pymc.gp`): `Latent`, `Marginal`, `MarginalApprox`,
`HSGP`, `HSGPPeriodic`, `TP` (Student-t process), `LatentKron`, `MarginalKron`. The split is:
`Latent`/`LatentKron`/`TP`/`HSGP` expose **`.prior(name, X)`** + **`.conditional(name, Xnew)`**;
`Marginal`/`MarginalApprox`/`MarginalKron` expose **`.marginal_likelihood(...)`** +
**`.conditional`** + **`.predict`** (and have **no** `.prior` method — a common confusion).

**Constructors take keyword args `mean_func` and `cov_func`** (defaults: `mean.Zero`,
`cov.Constant`). In v5 the constructors are keyword-only for these:

```python
import pymc as pm, numpy as np
gp = pm.gp.Marginal(mean_func=pm.gp.mean.Zero(), cov_func=cov)   # cov_func=, not positional
```

### 1. Marginal (GP regression, Gaussian likelihood) — VERIFIED
```python
X = np.linspace(0, 1, 50)[:, None]          # X MUST be 2-D, shape (n, 1) for 1-D input
with pm.Model() as model:
    ell = pm.InverseGamma("ell", alpha=2, beta=1)
    eta = pm.HalfNormal("eta", sigma=2)
    cov_func = eta**2 * pm.gp.cov.ExpQuad(input_dim=1, ls=ell)
    gp = pm.gp.Marginal(cov_func=cov_func)
    sigma = pm.HalfNormal("sigma", sigma=1)
    y_ = gp.marginal_likelihood("y", X=X, y=y, sigma=sigma)   # CURRENT arg is sigma=
    idata = pm.sample()

Xnew = np.linspace(-0.5, 1.5, 100)[:, None]
with model:
    f_pred = gp.conditional("f_pred", Xnew=Xnew)             # f only (latent function)
    # pred_noise=True gives f + noise (the predictive for *observations*)
    y_pred = gp.conditional("y_pred", Xnew=Xnew, pred_noise=True)
    idata.extend(pm.sample_posterior_predictive(idata, var_names=["f_pred", "y_pred"]))
```
Exact current signature (stable docs): `marginal_likelihood(name, X, y, sigma, jitter=1e-6,
is_observed=True, **kwargs)`. **`sigma=` is the current name**; older example notebooks pass
`noise=` (a `Covariance` or scalar) — both denote the additive observation noise, but writers
should use `sigma=` and flag `noise=` as the legacy spelling. `sigma` may be a scalar **or** a
`pm.gp.cov.*` Covariance for non-white/correlated noise. `.predict(Xnew, point=..., diag=...,
pred_noise=...)` returns numpy mean+cov given a single point (e.g. `idata...mean(...)`); use it
for fast point-estimate prediction without MCMC over `f_pred`.

### 2. Latent (non-Gaussian likelihoods) — VERIFIED from GP-Latent example
```python
with pm.Model() as model:
    ell = pm.InverseGamma("ell", mu=1.0, sigma=0.5)
    eta = pm.Exponential("eta", lam=1.0)
    cov = eta**2 * pm.gp.cov.ExpQuad(1, ell)
    gp = pm.gp.Latent(cov_func=cov)
    f = gp.prior("f", X=x[:, None])            # f is the latent function values at X
    # classification:
    p = pm.Deterministic("p", pm.math.invlogit(f))
    y_ = pm.Bernoulli("y", p=p, observed=y)
    # Poisson regression would be: lam = pm.math.exp(f); pm.Poisson("y", mu=lam, observed=y)
    idata = pm.sample(1000, chains=4)
with model:
    f_pred = gp.conditional("f_pred", Xnew, jitter=1e-4)   # bump jitter if Cholesky fails
```
`Latent.prior` internally **rotates `f` by the Cholesky factor** of the covariance (an automatic
non-centered reparameterization: `f = L @ v`, `v ~ Normal(0,1)`) to reduce posterior
correlations — this is why latent GPs sample at all. Args confirmed: `prior(name, X, reparameterize=True,
jitter=...)`, `conditional(name, Xnew, given=None, jitter=...)`.

### 3. MarginalApprox (sparse / inducing points) — VERIFIED
```python
Xu = pm.gp.util.kmeans_inducing_points(20, X)        # or fix / make Xu a free RV
gp = pm.gp.MarginalApprox(cov_func=cov, approx="VFE") # approx in {"VFE","FITC","DTC"}, default "VFE"
y_ = gp.marginal_likelihood("y", X=X, Xu=Xu, y=y, sigma=sigma)  # white IID noise only -> sigma scalar
```
Cuts O(n³) → O(n·m²) with m = #inducing points. VFE (Titsias 2009) is the default and usually
the safest; FITC can underestimate noise. Refs in docstring: Quiñonero-Candela & Rasmussen 2005;
Titsias 2009. (`MarginalSparse` was the **v3** name; v5 renamed it `MarginalApprox` — flag this.)

### 4. HSGP (Hilbert-space approximation) — VERIFIED, the modern scaling default
```python
gp = pm.gp.HSGP(m=[200], c=1.5, cov_func=cov)   # constructor: HSGP(m, L=None, c=None,
                                                #   drop_first=False, parametrization="noncentered",
                                                #   *, mean_func=Zero, cov_func)
f = gp.prior("f", X=x[:, None])
# ... likelihood on f ...
with model:
    f_pred = gp.conditional("f_pred", Xnew)
```
- **`m`** = number of basis functions per active dim (list). `m=[25,25]` ⇒ 625 basis fns. More
  `m` ⇒ captures **smaller** lengthscales, at cost.
- **`c`** = boundary extension factor; internally `L = c · S` with `S = max|X_centered|`. Larger
  `c` ⇒ captures **larger** lengthscales but typically needs larger `m`. Riutort-Mayol et al.
  recommend **c ≥ 1.2** minimum; data (train **and** predict) must sit **well inside** `[-L, L]`.
- **`L`** can be given directly instead of `c`.
- Helper (VERIFIED): `pm.gp.hsgp_approx.approx_hsgp_hyperparams(x_range, lengthscale_range,
  cov_func)` returns recommended `(m, c)`. Supported `cov_func` strings: `"expquad"`,
  `"matern52"`, `"matern32"`. Example: `x_range=[-5,95], lengthscale_range=[1,50],
  cov_func="matern52"` → `m≈543, c≈4.1`. There is also `gp.HSGP(...).set_boundary` internals
  (`L = c·S`). HSGP supports `ExpQuad`, `Matern52`, `Matern32` (and Periodic via the separate
  **`HSGPPeriodic`** class) — it does **not** support arbitrary kernels (needs a known spectral
  density). `parametrization="noncentered"` is the default; `"centered"` exists.

### 5. Covariance functions (`pm.gp.cov.*`) — VERIFIED names
`ExpQuad` (= RBF/squared-exponential/"exponentiated quadratic"), `RatQuad` (Rational Quadratic,
takes `alpha`, `ls`), `Matern12`, `Matern32`, `Matern52`, `Exponential`, `Periodic` (takes
`period`, `ls`), `Linear`, `Polynomial`, `Cosine`, `WhiteNoise(sigma)`, `Constant`,
`Gibbs` (nonstationary), `ScaledCov`, `Kron`, `Coregion`. Signature shape:
`pm.gp.cov.ExpQuad(input_dim, ls=..., ls_inv=..., active_dims=...)`; `Periodic(input_dim,
period, ls, active_dims=...)`. **Kernels compose**: `k1 + k2`, `k1 * k2`, `eta**2 * k` are all
valid (operator overloading returns new `Covariance` objects). Multiply amplitude in as
`eta**2 * cov`. Mean functions: `pm.gp.mean.Zero`, `Constant`, `Linear`.

### 6. Additive GPs & component extraction — VERIFIED
```python
gp1 = pm.gp.Marginal(cov_func=cov_trend)
gp2 = pm.gp.Marginal(cov_func=cov_seasonal)
gp  = gp1 + gp2                 # same class only
y_  = gp.marginal_likelihood("y", X=X, y=y, sigma=sigma)
# predict ONLY the trend component:
f_trend = gp1.conditional("f_trend", Xnew,
              given={"X": X, "y": y, "sigma": sigma, "gp": gp})
```
The `given=` dict is mandatory when predicting a sub-component, so the child GP knows the full
data/model context.

---

## Key concepts & equations the chapter must nail

- **GP definition**: a GP is a distribution over functions s.t. any finite set of inputs
  $X=\{x_i\}$ has $f(X)\sim \mathcal N(m(X), K(X,X))$. Fully specified by **mean function**
  $m(x)$ and **covariance/kernel** $k(x,x')$.
- **ExpQuad/RBF**: $k(x,x')=\eta^2\exp\!\big(-\frac{\|x-x'\|^2}{2\ell^2}\big)$. Infinitely
  differentiable ⇒ very smooth (often *too* smooth — a known criticism vs Matérn).
- **Matérn-ν**: $\nu=3/2$: $k=\eta^2(1+\frac{\sqrt3 r}{\ell})e^{-\sqrt3 r/\ell}$ (once diff'ble);
  $\nu=5/2$: $k=\eta^2(1+\frac{\sqrt5 r}{\ell}+\frac{5r^2}{3\ell^2})e^{-\sqrt5 r/\ell}$ (twice).
  Matérn-5/2 is the pragmatic default for "realistically rough" functions.
- **Periodic (ExpSineSquared)**: $k=\eta^2\exp\!\big(-\frac{2\sin^2(\pi|x-x'|/p)}{\ell^2}\big)$;
  `period` $p$ sets the cycle, `ls` the within-period wiggliness.
- **Linear / Polynomial / RatQuad**: Linear ⇒ Bayesian linear regression as a GP; RatQuad =
  scale-mixture of ExpQuads (extra `alpha` controls mixing of lengthscales).
- **Knob semantics (drill this)**: **lengthscale ℓ** = how far apart inputs must be before
  outputs decorrelate ⇒ controls *wiggliness*; small ℓ = wiggly/overfit, large ℓ ≈ flat.
  **amplitude η (eta)** = marginal SD of the function = vertical scale of deviations from the
  mean. **noise σ** = observation noise added on the diagonal (Marginal only).
- **GP regression posterior (Gaussian likelihood, closed form)** — the equation to show:
  $$\bar f_\* = K_{\*X}(K_{XX}+\sigma^2 I)^{-1}y,\qquad
    \mathrm{cov}(f_\*)=K_{\*\*}-K_{\*X}(K_{XX}+\sigma^2 I)^{-1}K_{X\*}.$$
  Marginal likelihood: $\log p(y\mid X)=-\tfrac12 y^\top(K+\sigma^2I)^{-1}y
   -\tfrac12\log|K+\sigma^2I|-\tfrac n2\log 2\pi$. The $(K+\sigma^2 I)^{-1}$ and $\log|\cdot|$
  are the **O(n³)** bottleneck (Cholesky of an n×n matrix).
- **Why InverseGamma on ℓ (Betancourt's robust GP)**: with finite data the likelihood is
  **flat / non-identified toward short lengthscales** — a GP can "explain" data by wiggling
  through every point (overfitting), which also creates nasty posterior geometry and divergences.
  An InverseGamma has **zero density at 0** (kills infinitesimal ℓ) and a **heavy right tail**
  (still permits large/smooth ℓ). Tune its two params so the prior puts ~1% mass **below** the
  smallest resolvable lengthscale (≈ typical spacing between data points) and ~1% **above** the
  data range (you can't learn structure longer than your domain). PyMC supports
  `pm.InverseGamma("ell", mu=..., sigma=...)` (mean/SD parameterization) or solve `alpha,beta`
  from the tail-probability constraints. Gamma priors are the common practical alternative (see
  Mauna Loa below) and also push mass off zero.
- **HSGP idea**: approximate the kernel by its **spectral density** $S(\omega)$ and a fixed set
  of Laplacian eigenfunctions on $[-L,L]$:
  $f(x)\approx\sum_{j=1}^m \sqrt{S(\sqrt{\lambda_j})}\,\phi_j(x)\,\beta_j$, $\beta_j\sim N(0,1)$.
  Cost drops to **O(nm + m)**; the basis is **independent of the kernel hyperparameters**, so it
  recomputes nothing during sampling. This is why HSGP is the go-to for medium/large 1-D (or
  low-D) problems.
- **GP↔spline↔BART connections**: Kimeldorf & Wahba (1970) — the posterior mean of a GP with a
  particular kernel **is** a smoothing spline (penalized regression on a basis). HSGP makes this
  explicit: a GP ≈ a basis-function (spline-like) regression with a principled prior on the
  coefficients. **BART** is the tree-based cousin: also a flexible nonparametric regression
  (sum of trees) with a regularizing prior, but it's adaptive/non-stationary and handles
  high-dim/interactions far better than a stationary GP, at the cost of GP's smooth uncertainty.
  Rule of thumb to state in Ch 14: **smooth low-D function with calibrated uncertainty ⇒ GP;
  many features / interactions / step-like structure ⇒ BART; known basis / 1-D wiggle ⇒ spline.**

---

## Common errors → causes → fixes

| Symptom | Cause | Fix |
|---|---|---|
| `ValueError`/shape error from `cov`/`marginal_likelihood` | `X` passed as 1-D `(n,)` | always reshape to `(n,1)`: `X[:, None]` |
| Cholesky / `LinAlgError: not positive definite` | near-duplicate inputs, tiny ℓ, no jitter | raise `jitter` (e.g. `1e-4`), increase ℓ prior lower bound, dedupe X, standardize inputs |
| Many divergences, ℓ wanders to ~0 | flat likelihood toward short lengthscales (non-identifiability) | use **InverseGamma/Gamma** ℓ prior (Betancourt), raise `target_accept` to 0.95–0.99 |
| `AttributeError: 'Marginal' object has no attribute 'prior'` | calling `.prior` on a Marginal/MarginalApprox | use `.marginal_likelihood`; `.prior` exists only on `Latent`/`HSGP`/`TP` |
| Latent GP samples horribly slow / stuck | n too large for O(n³) latent, or centered param | switch to **HSGP** or **MarginalApprox**; keep `reparameterize=True` |
| HSGP fit looks truncated/biased near edges | data near or outside `[-L,L]` | increase `c` (≥1.2, often 2–4); ensure predict points inside boundary |
| HSGP too smooth / misses wiggles | too few basis fns | increase `m`; use `approx_hsgp_hyperparams` to size `m,c` from lengthscale range |
| `noise=` "unexpected keyword" or deprecation | old v3/v4 spelling | use `sigma=` in `marginal_likelihood` (scalar or Covariance) |
| Predicted component nonsense in additive GP | forgot `given=` in sub-component `.conditional` | pass `given={"X":X,"y":y,"sigma":sigma,"gp":gp_full}` |
| `pred_noise` confusion | `.conditional` defaults `pred_noise=False` (returns latent f) | set `pred_noise=True` to get the predictive distribution **of observations** |

---

## Datasets & load snippets

- **Mauna Loa CO₂** (flagship for kernel combination, Ch 12 & 15):
  ```python
  import statsmodels.api as sm
  co2 = sm.datasets.co2.load_pandas().data          # weekly CO2 ppm, 1958–2001
  co2 = co2.dropna().resample("MS").mean().dropna()  # monthly
  # or sklearn: from sklearn.datasets import fetch_openml; fetch_openml("mauna-loa-atmospheric-co2", as_frame=True)
  ```
  Center x to years-since-start and standardize y; the canonical decomposition is
  **ExpQuad (long trend) + Periodic×ExpQuad (decaying seasonal) + RatQuad (medium) +
  Matern32+WhiteNoise (noise)**.
- **Synthetic 1-D GP draw** (best for teaching recovery of η, ℓ, σ):
  ```python
  rng = np.random.default_rng(8927)
  X = np.linspace(0, 10, 100)[:, None]
  true = 1.5**2 * np.exp(-0.5*(X-X.T)**2/1.0**2) + 1e-6*np.eye(100)
  f = rng.multivariate_normal(np.zeros(100), true)
  y = f + rng.normal(0, 0.3, 100)     # known truth: eta=1.5, ell=1.0, sigma=0.3
  ```
- **GP classification** (Latent + Bernoulli): synthetic logits via `invlogit`, as in GP-Latent.
- **Coal mining disasters** (`pm.get_data("coal.csv")`) works as a **Latent Poisson GP /
  log-intensity** example bridging to Ch 13/15.

---

## Curated resources (real URLs, with why + pointer)

- **PyMC `gp.Marginal` API** — https://www.pymc.io/projects/docs/en/stable/api/gp/generated/pymc.gp.Marginal.html — authoritative signature for `marginal_likelihood(name,X,y,sigma,...)`, `conditional`, `predict`.
- **PyMC `gp.Latent` API** — https://www.pymc.io/projects/docs/en/stable/api/gp/generated/pymc.gp.Latent.html — `.prior`/`.conditional`, the Cholesky reparameterization note.
- **PyMC `gp.HSGP` API** — https://www.pymc.io/projects/docs/en/stable/api/gp/generated/pymc.gp.HSGP.html — constructor `HSGP(m, L, c, drop_first, parametrization, *, cov_func)` and method sigs.
- **PyMC `gp.MarginalApprox` API** — https://www.pymc.io/projects/docs/en/stable/api/gp/generated/pymc.gp.MarginalApprox.html — VFE/FITC/DTC, `Xu` inducing points, O(nm²).
- **GP core notebook (overview of all classes)** — https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/Gaussian_Processes.html — which class has prior vs marginal_likelihood; additive `given=` pattern.
- **GP-Latent example** — https://www.pymc.io/projects/examples/en/latest/gaussian_processes/GP-Latent.html — verified runnable Bernoulli-logit & StudentT models; source of the InverseGamma(mu,sigma) ℓ + Exponential η pattern.
- **GP-Marginal example** — https://www.pymc.io/projects/examples/en/2022.01.0/gaussian_processes/GP-Marginal.html — clean regression walkthrough, `.conditional` with `pred_noise`.
- **HSGP Basic / First Steps** — https://www.pymc.io/projects/examples/en/latest/gaussian_processes/HSGP-Basic.html — rules of thumb for `m`,`c`; `approx_hsgp_hyperparams`; Gram-matrix validation.
- **HSGP Advanced Usage** — https://www.pymc.io/projects/examples/en/latest/gaussian_processes/HSGP-Advanced.html — multi-dim `m=[25,25]`, boundary intuition (`c=4` ⇒ boundary at 4× half-width).
- **Mauna Loa CO₂ (part 1)** — https://www.pymc.io/projects/examples/en/latest/gaussian_processes/GP-MaunaLoa.html — additive kernel design, component prediction.
- **Mauna Loa CO₂ (part 2)** — https://www.pymc.io/projects/examples/en/latest/gaussian_processes/GP-MaunaLoa2.html — full multi-component model, `ScaledCov` changepoint, Gamma ℓ priors (verbatim priors above).
- **Sparse Approximations example** — https://www.pymc.io/projects/examples/en/latest/gaussian_processes/GP-SparseApprox.html — inducing points, choosing `Xu`, VFE vs FITC.
- **Mean & Covariance Functions** — https://www.pymc.io/projects/examples/en/latest/gaussian_processes/GP-MeansAndCovs.html — visual catalog of every kernel + composition.
- **Betancourt, "Robust Gaussian Process Modeling"** — https://betanalpha.github.io/assets/case_studies/gaussian_processes.html — the canonical reference for ℓ non-identifiability and the InverseGamma fix (the *why* behind penalizing short lengthscales).
- **Rasmussen & Williams, *GPML* (free full PDF)** — https://gaussianprocess.org/gpml/chapters/RW.pdf — Ch 2 (regression, predictive eqns), Ch 4 (covariance functions), Ch 6.3 (GP↔spline). The textbook.
- **Riutort-Mayol et al. 2022, HSGP for PPLs** — https://arxiv.org/abs/2004.11408 — the m/c/L theory; cite for HSGP design choices.
- **Solin & Särkkä 2019, Hilbert-space reduced-rank GP** — the original HSGP method (Stat. Comput.) — basis derivation behind `pm.gp.HSGP`.
- **Stan User's Guide, Fitting GPs** — https://mc-stan.org/docs/stan-users-guide/gaussian-processes.html — the Stan-side `cov_exp_quad`/`gp_*` view and the same InverseGamma ℓ advice; good for the Stan cross-reference.
- **Juan Orduz, GP time-series with PyMC** — https://juanitorduz.github.io/gp_ts_pymc3/ — practitioner blog, additive kernels for forecasting (update APIs to v5 when citing).

---

## Uncertainties / things the writer should double-check

- **`approx="VFE"` default** for `MarginalApprox`: I confirmed VFE/FITC/DTC are the options and VFE
  is documented as default — re-confirm the exact default string against the installed version.
- **`Periodic` argument order**: docs show `Periodic(input_dim, period, ls, active_dims=...)`;
  confirm whether `period` is positional or keyword in the installed version (`period=` is safest).
- **`pm.gp.util.kmeans_inducing_points`**: I cited it as the inducing-point helper from the sparse
  example; verify the exact import path (`pm.gp.util.kmeans_inducing_points`) in v5.27 — it may be
  `pm.gp.util.kmeans_inducing_points(n_inducing, X)` arg order.
- **Betancourt exact InverseGamma numbers**: the case-study page is too large to fetch cleanly; the
  *method* (tune tails to ~1% below min-spacing, ~1% above data range) is well-attested, but quote
  any specific `alpha,beta` only after reading the case study or use PyMC's `InverseGamma(mu,sigma)`
  + a solver. The PreliZ `pz.maxent`/`pz.InverseGamma(...).to_pymc()` workflow is a clean way to set
  these from interval constraints if PreliZ is in scope.
- **`HSGPPeriodic`**: confirmed it exists for periodic HSGP; double-check its constructor args
  (`m`, `scale`, `cov_func=Periodic(...)`) before writing code, as they differ from plain `HSGP`.
- **Mauna Loa load**: `statsmodels` `sm.datasets.co2` is reliable; the sklearn OpenML route name can
  change — prefer statsmodels or the CSV bundled in `pymc-examples`.
- **`.predict` return**: it returns numpy mean+cov (or var with `diag=True`) for a *single* posterior
  point, not an InferenceData — make sure example code passes a `point` dict (e.g.
  `az.extract(idata).mean(...)`-derived) and doesn't conflate it with `.conditional`.
