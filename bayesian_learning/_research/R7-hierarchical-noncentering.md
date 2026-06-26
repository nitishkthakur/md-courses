# R7 — Hierarchical / Multilevel Models & Non-Centering (feeds Chapter 10)

Research dossier for the chapter writer. Verified against current PyMC v5 / Bambi / ArviZ
docs (June 2026). Where I could not verify an exact symbol I say so explicitly. This is
dense reference material, not prose.

---

## Scope & which chapters use this

Primary consumer: **Chapter 10 — Hierarchical & Multilevel Models**. The non-centered
parameterization material is *introduced conceptually* in Ch 5 (HMC geometry) and Ch 7
(divergences / funnel), then **applied as the default cure in Ch 10** — keep terminology
identical to those chapters (course bible §10). Eight-schools funnel is shared with Ch 7.
The radon flagship and `LKJCholeskyCov` correlated-effects pattern resurface in the Ch 18
capstone. Bambi formula syntax for multilevel models echoes Ch 9 (GLMs/Bambi).

Topics this dossier must equip the writer to nail: complete vs no vs partial pooling;
varying intercepts and slopes; the radon example (canonical Gelman/Hill setup + exact
data load); shrinkage and the bias–variance reading; hyperpriors on group means and group
scales; the funnel pathology and **non-centered parameterization in depth** (offset trick,
when centered is actually better); correlated varying effects via `LKJCholeskyCov` +
`MvNormal`; predicting for new/unseen groups; `ZeroSumNormal` for identifiability.

---

## Verified API & idioms (confirmed against docs)

**`pm.LKJCholeskyCov`** — verified signature (PyMC stable docs):
```python
pm.LKJCholeskyCov(name, eta, n, sd_dist, *, compute_corr=True, store_in_trace=True, **kwargs)
```
- `eta > 0` (LKJ shape; `eta=1` ⇒ uniform over correlation matrices, larger ⇒ shrinks toward identity/zero correlation).
- `n > 1` is the matrix dimension (e.g. `n=2` for intercept+slope).
- `sd_dist` **must be an *unregistered* distribution built with the `.dist()` API**, e.g. `pm.Exponential.dist(1.0, shape=2)` or `pm.HalfNormal.dist(sigma=np.array([50,10]), shape=2)`. It is *cloned* internally (warning in docs: "sd_dist will be cloned, rendering it independent of the one passed as input"). Do NOT pass a normal RV like `pm.Exponential("sd", 1)`.
- With `compute_corr=True` (the default in v5) it returns a **3-tuple `chol, corr, stds`** (unpacked Cholesky factor `L`, correlation matrix, std-dev vector). With `compute_corr=False` it returns only the packed Cholesky. **This is a v3→v5 change** — old v3 code returned the packed Cholesky and you called `pm.expand_packed_triangular`. In v5 prefer the 3-tuple.
- `store_in_trace=False` keeps `corr`/`stds` out of the InferenceData (smaller files).

**`pm.MvNormal` / Cholesky pattern.** Two equivalent ways to get correlated varying effects:
```python
# (a) MvNormal directly with the Cholesky factor:
ab = pm.MvNormal("ab", mu=mu_ab, chol=chol, dims=("county", "param"))   # centered
# (b) NON-CENTERED via the offset trick on the Cholesky factor (preferred for funnels):
z   = pm.Normal("z", 0.0, 1.0, dims=("param", "county"))
ab  = pm.Deterministic("ab", (mu_ab[:, None] + pt.dot(chol, z)).T, dims=("county","param"))
```
`pt` is `import pytensor.tensor as pt` (v5; **not** `theano.tensor` and **not** `aesara`). The PyMC radon primer uses `pt.dot(chol, z).T`.

**`pm.ZeroSumNormal`** — verified it exists in v5 (`pymc.ZeroSumNormal`). Signature per docs:
```python
pm.ZeroSumNormal(name, sigma=1, *, n_zerosum_axes=1, dims=None, shape=None)
```
A Normal whose specified axes are constrained to sum to zero (last axis by default).
Constraint from docs: **"sigma cannot have length > 1 across the zero-sum axes."** Use it
for the varying-effect deflections so the global intercept is identifiable (the group
offsets sum to zero rather than being free to trade off against `mu_a`). Example:
```python
with pm.Model(coords={"county": mn_counties}) as m:
    a_bar = pm.Normal("a_bar", 0, 1)
    a_dev = pm.ZeroSumNormal("a_dev", sigma=sigma_a, dims="county")  # sums to 0 over county
    alpha = a_bar + a_dev
```

**`coords` / `dims` idiom (v5, use throughout).**
```python
coords = {"county": mn_counties, "obs_id": np.arange(len(log_radon)), "param": ["alpha","beta"]}
with pm.Model(coords=coords) as model:
    county_idx = pm.Data("county_idx", county, dims="obs_id")   # mutable by default in v5
    alpha = pm.Normal("alpha", mu_a, sigma_a, dims="county")
    y = pm.Normal("y", alpha[county_idx], sigma=sigma_y, observed=log_radon, dims="obs_id")
```
`pm.Data` is mutable by default in v5 — the v4 `mutable=True` kwarg is deprecated/removed;
do not pass it. Swap data with `pm.set_data({"county_idx": new_idx})` before
`pm.sample_posterior_predictive`.

**Non-centered scalar offset trick (the canonical Ch 10 form), verified verbatim from the PyMC radon primer:**
```python
z_a   = pm.Normal("z_a", mu=0, sigma=1, dims="county")
alpha = pm.Deterministic("alpha", mu_a + z_a * sigma_a, dims="county")
```

**Bambi multilevel formula syntax** (verified from the Bambi radon notebook):
```python
import bambi as bmb
# varying intercept only:
m = bmb.Model("log_radon ~ floor + (1|county)", data=df)
# varying intercept + varying slope of floor, correlated by default:
m = bmb.Model("log_radon ~ floor + (floor|county)", data=df)
# uncorrelated varying intercept + slope:
m = bmb.Model("log_radon ~ 0 + floor + (0 + floor|county)", data=df)
idata = m.fit(draws=1000, tune=1000, target_accept=0.9)   # returns az.InferenceData
```
Bambi has a `noncentered=True` default (pass `noncentered=False` to force centered).
Always also show the raw-PyMC equivalent (bible §10).

**Sampler call (consistent with bible §10):**
`idata = pm.sample(1000, tune=1000, chains=4, target_accept=0.9, random_seed=RANDOM_SEED)`;
raise `target_accept` to 0.95–0.99 when discussing residual divergences.

---

## Key concepts & equations the chapter must nail

**The three pooling regimes.** For groups $j=1..J$ with observations $y_{ij}$:
- *Complete pooling*: one shared $\alpha$ — ignores group structure (high bias, low variance).
- *No pooling*: independent $\alpha_j$ per group — overfits small groups (low bias, high variance).
- *Partial pooling (hierarchical)*: $\alpha_j \sim \mathrm{Normal}(\mu_\alpha, \sigma_\alpha)$ — the data *learn* how much to pool. This is the whole point.

**Generative form (varying intercept).**
$$ y_{ij} \sim \mathrm{Normal}(\alpha_{j[i]} + \beta x_{ij},\ \sigma_y),\quad \alpha_j \sim \mathrm{Normal}(\mu_\alpha,\ \sigma_\alpha) $$
with hyperpriors $\mu_\alpha \sim \mathrm{Normal}(0, 10)$, $\sigma_\alpha \sim \mathrm{Exponential}(1)$ (or $\mathrm{HalfNormal}$ / $\mathrm{HalfCauchy}(5)$; cite Gelman 2006 for variance-param priors — avoid `InverseGamma(0.001,0.001)`).

**Shrinkage / partial pooling weight (the bias–variance reading).** For a Gaussian group
mean with $n_j$ observations, the posterior mean is a precision-weighted average:
$$ \hat\alpha_j \approx \frac{\frac{n_j}{\sigma_y^2}\,\bar y_j + \frac{1}{\sigma_\alpha^2}\,\mu_\alpha}{\frac{n_j}{\sigma_y^2} + \frac{1}{\sigma_\alpha^2}} $$
So small-$n_j$ groups shrink hard toward $\mu_\alpha$; large-$n_j$ groups barely move.
This is exactly a Bayesian/empirical-Bayes reading of the **James–Stein estimator**:
shrinkage trades a little bias for a large variance reduction and *dominates* the unpooled
MLE in total squared error when $J \ge 3$. Stein (1962) gave the hierarchical/empirical-Bayes
interpretation. The chapter should plot the classic "shrinkage" figure: unpooled estimates
vs partially-pooled estimates, with arrows pulling small groups toward the grand mean.

**Varying slopes + correlation.** Stack intercept and slope per group as a 2-vector and
give it a multivariate normal population:
$$ \begin{bmatrix}\alpha_j\\\beta_j\end{bmatrix} \sim \mathrm{MvNormal}\!\left(\begin{bmatrix}\mu_\alpha\\\mu_\beta\end{bmatrix},\ \Sigma\right),\quad \Sigma = \mathrm{diag}(\sigma)\,R\,\mathrm{diag}(\sigma) = LL^\top $$
$R$ gets an LKJ prior; $L$ is its Cholesky factor times the std-devs. Modeling the
correlation lets the model borrow strength *across* parameters (e.g. high-intercept
counties also tend to have steeper slopes).

**The funnel.** In the centered parameterization the joint posterior of
$(\alpha_j, \log\sigma_\alpha)$ is funnel-shaped: when $\sigma_\alpha$ is small the
$\alpha_j$ are squeezed into a narrow neck; when $\sigma_\alpha$ is large they spread out.
The local curvature changes by orders of magnitude along the funnel, so a *single* HMC
step size cannot fit both the wide mouth and the narrow neck. Step sizes tuned for the
mouth overshoot in the neck → the leapfrog integrator diverges. **Divergences cluster in
the neck (small $\sigma_\alpha$)** — they are truth-tellers (Betancourt), not noise.

**Non-centered reparameterization (the cure).** Replace the direct draw with a
standardized latent variable plus an affine map:
$$ \tilde\alpha_j \sim \mathrm{Normal}(0,1),\qquad \alpha_j = \mu_\alpha + \sigma_\alpha\,\tilde\alpha_j. $$
Now $\tilde\alpha_j$ and $\sigma_\alpha$ are *a priori independent* — the funnel is "unrolled"
into a roughly isotropic Gaussian that HMC explores efficiently. This is the "Matt trick"
(named for Matt Hoffman on stan-users). Same idea for the Cholesky case:
$\mathbf{ab}_j = \mu_{ab} + L\,\mathbf{z}_j$ with $\mathbf{z}_j \sim \mathrm{Normal}(0, I)$.

**When centered is actually BETTER (high-data regime).** This is the nuance the chapter
must include and not get backwards. The optimal parameterization is governed by **whether
the prior or the likelihood dominates each group's posterior** (Betancourt & Girolami):
- *Likelihood dominates* (many observations per group, large $\sigma_\alpha$) ⇒ **centered**
  is more efficient — the posterior is already well-conditioned and the non-centering
  introduces its own awkward geometry.
- *Prior dominates* (few observations per group, small $\sigma_\alpha$, sparse data) ⇒
  **non-centered** wins decisively.
The honest practitioner rule: default to non-centered for sparse/weakly-informed groups,
but in big-data-per-group settings test the centered form — it may sample faster with
higher ESS. With heterogeneous group sizes you can even **mix**: centered for data-rich
groups, non-centered for data-poor ones.

**Identifiability with `ZeroSumNormal`.** A varying-intercept model with both a global
$\mu_\alpha$ AND free $\alpha_j$ has a soft non-identifiability: you can add a constant to
$\mu_\alpha$ and subtract it from every $\alpha_j$. Hierarchical priors tame this, but
constraining the deflections to sum to zero (`ZeroSumNormal`) makes the global level
crisply interpretable and often improves sampling — especially for categorical
fixed-effect-style factors with no natural baseline.

---

## Common errors → causes → fixes

| Symptom | Cause | Fix |
|---|---|---|
| Many divergences, clustered at small `sigma_a` | Centered funnel: curvature too high in the neck | Switch to **non-centered** (`z`→`mu + z*sigma`); only then raise `target_accept` to 0.95–0.99 |
| Divergences persist after non-centering | Step size still too large; very tight neck | `target_accept=0.99`, more `tune`; check for a second funnel (e.g. correlated slopes also need non-centering) |
| `sigma_a` posterior piles up at 0; low ESS | Centered sampler can't reach the neck — *misses* low-variance solutions | Non-center; inspect `az.plot_pair` of `sigma_a` vs a group param colored by divergences |
| `LKJCholeskyCov` raises about `sd_dist` | Passed a *registered* RV instead of a `.dist()` object | Use `pm.Exponential.dist(1.0, shape=n)` (note `.dist`), never `pm.Exponential("sd", 1)` |
| `ValueError`/shape error unpacking LKJ | Assumed packed Cholesky (v3 habit) | In v5 `compute_corr=True` returns `chol, corr, stds` — unpack 3 values |
| `pm.Data(..., mutable=True)` errors / warns | `mutable` kwarg removed; Data is mutable by default in v5 | Drop the kwarg; use `pm.set_data(...)` to swap |
| Posterior predictive for new group errors on shape/coords | Coord length changed but variable shape didn't | Use `pm.set_data` with matching new index AND update coords; or draw fresh offsets `z_new ~ Normal(0,1, dims="new_group")` and recompute deterministically; see PyMC-Labs out-of-model post |
| Wrong county→row mapping (silent bug) | Index built from a non-contiguous / unsorted factor | Build `county_lookup = dict(zip(mn_counties, range(len(mn_counties))))` and `county = df.county.replace(county_lookup).values`; verify `alpha[county_idx]` lines up |
| `HalfCauchy`/`Uniform(0,100)` on `sigma_y` gives heavy tails / slow mixing | Diffuse scale priors | Prefer `Exponential(1)` or `HalfNormal`; cite Gelman 2006; warn against `InverseGamma(ε,ε)` |
| Precision vs sd confusion | BUGS/JAGS used precision $\tau=1/\sigma^2$ | PyMC/Stan use **sd $\sigma$** — state this loudly (bible §5) |

---

## Datasets & load snippets

**Radon (Minnesota) — the flagship.** Ships with PyMC via `pm.get_data`. Verified
preprocessing (Gelman/Hill canonical setup):
```python
import pymc as pm, pandas as pd, numpy as np
srrs2 = pd.read_csv(pm.get_data("srrs2.dat"))
srrs2.columns = srrs2.columns.map(str.strip)
srrs_mn = srrs2[srrs2.state == "MN"].copy()
srrs_mn["fips"] = srrs_mn.stfips * 1000 + srrs_mn.cntyfips

cty = pd.read_csv(pm.get_data("cty.dat"))
cty_mn = cty[cty.st == "MN"].copy()
cty_mn["fips"] = 1000 * cty_mn.stfips + cty_mn.ctfips

srrs_mn = srrs_mn.merge(cty_mn[["fips", "Uppm"]], on="fips").drop_duplicates(subset="idnum")
srrs_mn.county = srrs_mn.county.map(str.strip)
mn_counties   = srrs_mn.county.unique()                       # 85 counties
county_lookup = dict(zip(mn_counties, range(len(mn_counties))))
county        = srrs_mn["county_code"] = srrs_mn.county.replace(county_lookup).values
log_radon     = np.log(srrs_mn.activity + 0.1).values         # +0.1 avoids log(0)
floor_measure = srrs_mn.floor.values                          # 0 = basement, 1 = first floor
u             = np.log(srrs_mn.Uppm.values)                    # county-level uranium predictor
```
`floor=1` (ground floor) ⇒ *less* radon (negative slope); `u` (log uranium) is a
**group-level predictor** that enters the *intercept* model, illustrating how covariates
can live at either level. Alt source: `rmcelreath`/Gelman ARM repo
(`stat.columbia.edu/~gelman/arm/examples/radon`) or the `pymc-devs/pymc4` `radon.csv`.

**Eight Schools** (shared with Ch 7, the minimal funnel demo):
```python
J = 8
y     = np.array([28., 8., -3., 7., -1., 1., 18., 12.])
sigma = np.array([15., 10., 16., 11., 9., 11., 10., 18.])
```

**Sleepstudy** (lme4 classic; clean varying intercept+slope with real correlation) — load
from the `lme4`/`rpy2` ports or a CSV; columns `Reaction ~ Days + (Days|Subject)`, 18 subjects.

**Chimpanzees / reedfrogs / Bangladesh fertility** (rethinking repo raw CSVs) for
hierarchical logistic / binomial varying effects — used in McElreath Ch 12–14 ports.

---

## Curated resources (each with a real URL + why + pointer)

- **PyMC — A Primer on Bayesian Methods for Multilevel Modeling (radon).** https://www.pymc.io/projects/examples/en/latest/generalized_linear_models/multilevel_modeling.html — *The* canonical PyMC v5 radon notebook: complete/no/partial pooling, varying intercept & slope, non-centered offset trick, and the `LKJCholeskyCov` + `pt.dot(chol, z).T` covariation model. Copy the data-load and the `covariation_intercept_slope` block from here.
- **Thomas Wiecki — "Why hierarchical models are awesome, tricky, and Bayesian."** https://twiecki.io/blog/2017/02/08/bayesian-hierchical-non-centered/ — Best plain-language funnel explanation and the centered↔non-centered code (`b_offset`, `b = mu_b + b_offset*sigma_b`). Use its funnel scatter + reparameterized-funnel plots as the chapter's mental model.
- **Betancourt — "Hierarchical Modeling" case study.** https://betanalpha.github.io/assets/case_studies/hierarchical_modeling.html — Rigorous theory: prior-vs-likelihood domination determines parameterization, funnel geometry, divergences as truth-tellers. (Page is large; the PDF mirror is at betanalpha.github.io/assets/case_studies/hierarchical_modeling.pdf.)
- **Betancourt — "Diagnosing Biased Inference with Divergences."** https://www.pymc.io/projects/examples/en/latest/diagnostics_and_criticism/Diagnosing_biased_Inference_with_Divergences.html — PyMC port; shows divergences clustering in the neck and the non-centered fix, plus `target_accept` tuning. Pairs with Ch 7.
- **Betancourt & Girolami — "Hamiltonian Monte Carlo for Hierarchical Models" (2015), arXiv:1312.0906.** https://arxiv.org/abs/1312.0906 — The primary source for "centered when likelihood dominates, non-centered when prior dominates." Cite for the high-data-regime nuance.
- **PyMC — Statistical Rethinking Lecture 12: Multilevel Models.** https://www.pymc.io/projects/examples/en/latest/statistical_rethinking_lectures/12-Multilevel_Models.html — McElreath Ch 13 port (reedfrogs, varying intercepts, shrinkage plots).
- **PyMC — Statistical Rethinking Lecture 13: Multilevel Adventures.** https://www.pymc.io/projects/examples/en/latest/statistical_rethinking_lectures/13-Multilevel_Adventures.html — McElreath Ch 14 territory (Bangladesh fertility, non-centered varying effects `alpha_bar + z*sigma`). Note: full correlated-effects MvNormal pattern is deferred here — use the PyMC radon primer for that.
- **Tomi Capretto — "Hierarchical modeling with the LKJ prior in PyMC."** https://tomicapretto.com/posts/2022-06-12_lkj-prior/ — Cleanest worked `LKJCholeskyCov(..., compute_corr=True, store_in_trace=False)` → `MvNormal`/non-centered `at.dot(L, u_raw).T` on sleepstudy, with `sd_dist = pm.HalfNormal.dist(...)`. Best template for correlated varying slopes.
- **Bambi — Hierarchical Linear Regression (radon).** https://bambinos.github.io/bambi/notebooks/radon_example.html — Verified formula syntax `(1|county)`, `(floor|county)`, prior dict, `noncentered=` flag, `.fit()`→InferenceData. Use for the "and here's the one-liner" Bambi panel.
- **PyMC docs — `pymc.LKJCholeskyCov`.** https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.LKJCholeskyCov.html — Verified signature, `sd_dist` `.dist()` requirement, 3-tuple return.
- **PyMC docs — `pymc.ZeroSumNormal`.** https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.ZeroSumNormal.html — Verified `n_zerosum_axes`, the "sigma length ≤ 1 across zero-sum axes" constraint, `dims` example.
- **PyMC Labs — "Out-of-Model Predictions in PyMC."** https://www.pymc-labs.com/blog-posts/out-of-model-predictions-with-pymc — How to predict for *new/unseen groups*: instantiate fresh offsets from the population, pass via `set_data`/`var_names` to `sample_posterior_predictive`. Pair with the Alex Jones case study: https://alexjonesphd.github.io/Out-Of-Sample/
- **Stan User's Guide — Reparameterization.** https://mc-stan.org/docs/stan-users-guide/reparameterization.html — The reference statement of the non-centered ("Matt trick") parameterization; cross-language confirmation of the same algebra.
- **Gelman (2006) — "Prior distributions for variance parameters in hierarchical models."** (Bayesian Analysis 1(3)) — Cite for `HalfNormal`/`HalfCauchy`/`Exponential` on group scales and the warning against `InverseGamma(ε,ε)`.

---

## Predicting for new/unseen groups (the pattern to teach)

Two correct routes, both worth showing:
1. **Population draw (new group, no data yet).** For a brand-new county, its intercept is a
   fresh draw from the *learned* population: `alpha_new = mu_a + Normal(0,1)*sigma_a`. In
   PyMC, add a coord for the new group, define `z_new ~ Normal(0,1, dims="new")`, compute
   the deterministic, and `pm.sample_posterior_predictive(idata, var_names=[...],
   predictions=True)`. The posterior of `sigma_a` automatically inflates predictive
   uncertainty for unseen groups — the headline advantage of the hierarchical model.
2. **Swap-in via `pm.set_data`.** Wrap indices/predictors in `pm.Data`; for new *rows of
   existing groups* just `pm.set_data({"county_idx": new_idx, "floor_idx": new_floor})`
   and resample the predictive. For genuinely new groups, the variable's `dims` length must
   change too — update the coord (route 1 is cleaner and less error-prone).

⚠️ The matching of model variables to posterior samples in
`sample_posterior_predictive` is **by name** — keep names/shapes/coords compatible across
the fitting model and the prediction model.

---

## Uncertainties / things the writer should double-check

- **`pt.dot(chol, z).T` orientation.** I verified the PyMC radon primer uses `pt.dot(chol, z).T`
  with `z` of dims `("param","county")` and Capretto uses `at.dot(L, u_raw).T` with
  `("effect","subject")`. The transpose and which axis is the matrix dimension are easy to
  get wrong — the writer should run a tiny shape check (`L` is `(2,2)`, `z` is
  `(2, n_groups)` ⇒ product `(2, n_groups)` ⇒ `.T` ⇒ `(n_groups, 2)`). Capretto's post
  still imports the old alias `at` (aesara/`at.dot`); in current v5 this is `pt`
  (`pytensor.tensor`). Update the alias when transcribing.
- **`ZeroSumNormal` `sigma` as a vector.** Docs forbid `sigma` length > 1 across the
  zero-sum axis. If you want per-group scales you cannot put them on the zero-sum axis —
  double-check before writing a vectorized-scale example.
- **`store_in_trace` vs `compute_corr` exact defaults** can drift between minor v5 releases
  (I confirmed `compute_corr=True`, `store_in_trace=True` on the stable docs page). Add a
  `# verify in current PyMC docs` comment if pinning matters.
- **Bambi `noncentered` default** — confirmed the kwarg exists and the radon notebook
  passes `noncentered=False` to demo the centered form; verify the *default* value (it is
  `True`) against the installed Bambi version.
- **Radon county count.** The Gelman/Hill canonical setup yields **85** Minnesota counties;
  some derived CSVs differ slightly after dedup. Verify `len(mn_counties)` in the actual run.
- **`pm.get_data("srrs2.dat")` / `"cty.dat"`** ship with PyMC — confirmed widely used, but
  if a stripped install lacks them, fall back to the Gelman ARM repo or `pymc4/radon.csv`.
- Betancourt hierarchical case study page exceeded the fetch size limit here; I sourced its
  centered-vs-non-centered claim from the arXiv paper (1312.0906) and the PyMC divergences
  port. The writer should read the case study directly for the geometry figures.
