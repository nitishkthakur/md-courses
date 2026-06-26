# R3 — MCMC From Metropolis to NUTS (research dossier)

Reference material for chapter authors. Verified against PyMC v5 / ArviZ docs (June 2026). Not prose; dense reference.

## Scope & which chapters use this

- **Primary: Chapter 05** `05-inference-2-mcmc-from-metropolis-to-nuts.md`. Full arc: Markov chains → detailed balance → Metropolis–Hastings → Gibbs → why random-walk MCMC fails in high-D (typical set / concentration of measure, Betancourt framing) → Hamiltonian Monte Carlo (physical analogy, momentum, Hamiltonian, leapfrog) → NUTS (no-U-turn criterion) → warmup/adaptation (dual averaging step size, mass/metric matrix, target_accept, max_treedepth) → what each `pm.sample()` knob does.
- **Supports Chapter 07** (`07-mcmc-diagnostics-and-debugging.md`): divergences are the truth-tellers HMC gives you; energy/BFMI; the funnel; centered vs non-centered. Ch 05 introduces these conceptually, Ch 07 goes deep on reading them.
- **Supports Chapter 17** (`17-scaling-and-advanced-computation.md`): `nuts_sampler=` backends (numpyro/nutpie/blackjax), why gradients + JAX matter for scale.

This dossier deliberately overlaps R5 (diagnostics) at the boundary — keep the *mechanism* (why divergences happen, what the energy is) in Ch 05; defer the *diagnostic loop* (thresholds, full ArviZ workflow) to Ch 07/R5.

## Verified API & idioms (confirmed against PyMC v5 docs)

**`pm.sample()` full signature** (verified `pymc.io/.../pymc.sample.html`, PyMC 5.26.x / "stable"). The default sampler is NUTS for continuous models; only `draws` is positional, everything else is keyword-only (`*`):

```python
pm.sample(
    draws=1000, *,
    tune=None,                 # docstring: "Number of iterations to tune, defaults to 1000."
    chains=None,               # = max(cores, 2)
    cores=None,                # = min(n_cpus, 4)
    random_seed=None,
    progressbar=True,
    step=None,                 # auto-assigned if None
    var_names=None,
    nuts_sampler=None,         # legacy alias resolves to "pymc"; one of
                               #   ["pymc","nutpie","blackjax","numpyro"]
    initvals=None,
    init="auto",               # NOTE default is "auto" -> resolves to "jitter+adapt_diag"
    jitter_max_retries=10,
    n_init=200000,             # only used by ADVI-based inits
    trace=None,
    discard_tuned_samples=True,
    compute_convergence_checks=True,
    return_inferencedata=True, # returns az.InferenceData
    idata_kwargs=None,
    nuts_sampler_kwargs=None,
    callback=None,
    mp_ctx=None,
    blas_cores="auto",
    model=None,
    **kwargs                   # extra kwargs forwarded to the step method's __init__
)
```

**Where `target_accept` / `max_treedepth` go.** They are NUTS step-method kwargs, NOT named top-level params. Two equivalent idioms (verified):

```python
# (a) most common: forwarded via **kwargs to the auto-assigned NUTS
idata = pm.sample(1000, tune=1000, chains=4, target_accept=0.9, random_seed=RANDOM_SEED)

# (b) explicit nuts={...} dict
idata = pm.sample(1000, tune=1000, nuts={"target_accept": 0.95, "max_treedepth": 12})

# (c) hand-built step method (most explicit; used when mixing samplers)
with model:
    step = pm.NUTS(target_accept=0.95, max_treedepth=12)
    idata = pm.sample(1000, tune=1000, step=step)
```
`max_treedepth` default is **10** (so up to 2^10 = 1024 leapfrog steps per draw). `target_accept` default is **0.8** for the pymc NUTS.

**`init` options** (verified `pmc.init_nuts`): `"auto"` (→`jitter+adapt_diag`), `"adapt_diag"`, `"jitter+adapt_diag"` (default), `"adapt_full"`, `"jitter+adapt_full"`, `"advi"`, `"advi+adapt_diag"`, `"advi_map"`, `"map"`, `"jitter+adapt_diag_grad"`. Meaning of the metric ones:
- `adapt_diag`: start from identity mass matrix, adapt a **diagonal** metric from the variance of tuning samples.
- `jitter+adapt_diag`: same but add uniform jitter in [-1,1] to each chain's start (so chains start dispersed → R-hat is meaningful). **This is the default.**
- `adapt_full` / `jitter+adapt_full`: adapt a **dense** mass matrix from the sample covariance of tuning draws (captures posterior correlations; more expensive, ~O(d²) per step). `init` is ignored when you pass `step=` manually.

**Step methods that exist in v5** (verified): `pm.NUTS`, `pm.HamiltonianMC`, `pm.Metropolis`, `pm.BinaryMetropolis`, `pm.BinaryGibbsMetropolis`, `pm.CategoricalGibbsMetropolis`, `pm.DEMetropolis`, `pm.DEMetropolisZ`, `pm.Slice`. PyMC auto-assigns NUTS to continuous vars and a Gibbs/Metropolis variant to discrete vars, forming a **CompoundStep**. There is no standalone `pm.Gibbs` class — Gibbs in PyMC appears as the `*GibbsMetropolis` discrete steppers; teach Gibbs as *concept* + show conjugate custom step.

**Accessing sampler diagnostics from InferenceData** (verified, eight-schools divergence notebook):
```python
divergent = idata.sample_stats["diverging"]          # boolean DataArray (chain, draw)
n_div = int(divergent.sum())                          # count divergences
pct   = float(divergent.mean()) * 100                 # percentage
# other sample_stats keys: energy, step_size, tree_depth, n_steps,
#   acceptance_rate (a.k.a. mean_tree_accept), lp, max_energy_error, process_time_diff
az.plot_energy(idata)                                 # marginal vs transition energy + BFMI
az.plot_pair(idata, var_names=["theta","tau"], divergences=True)  # locate divergences
```
(Note: very recent ArviZ uses `visuals={"divergence": True}` on `plot_pair`; older/stable uses `divergences=True`. Flag both for the writer.)

**Non-centered reparameterization** (the canonical cure, verified spelling):
```python
with pm.Model() as noncentered:
    mu  = pm.Normal("mu", 0, 5)
    tau = pm.HalfCauchy("tau", 5)
    theta_t = pm.Normal("theta_t", 0, 1, shape=J)         # standardized
    theta   = pm.Deterministic("theta", mu + tau*theta_t) # rescale
    pm.Normal("obs", mu=theta, sigma=sigma, observed=y)
```

**Backends** (Ch 17 hook): `pm.sample(nuts_sampler="numpyro")`, `"nutpie"`, `"blackjax"`. These run JAX/Rust NUTS but accept the same `target_accept`, `tune`, `draws`, `chains`; pass backend-specific options via `nuts_sampler_kwargs={...}`. nutpie default init differs (it does its own warmup schedule).

## Key concepts & equations the chapter must nail

**Markov chain + invariance.** A transition kernel $T(\theta'\mid\theta)$ with the target $p$ as its stationary distribution: $\int T(\theta'\mid\theta)\,p(\theta)\,d\theta = p(\theta')$. We want the chain to be irreducible + aperiodic so the ergodic theorem gives $\frac{1}{N}\sum_n f(\theta_n) \to \mathbb{E}_p[f]$.

**Detailed balance (sufficient, not necessary, for invariance).**
$$ p(\theta)\,T(\theta'\mid\theta) = p(\theta')\,T(\theta\mid\theta'). $$
Detailed balance ⇒ $p$ is stationary. Reversibility is sufficient; HMC actually uses a more subtle (volume-preserving, time-reversible) argument — make the distinction explicit.

**Metropolis–Hastings.** Propose $\theta' \sim q(\theta'\mid\theta)$, accept with
$$ \alpha = \min\!\Big(1,\ \frac{p(\theta')\,q(\theta\mid\theta')}{p(\theta)\,q(\theta'\mid\theta)}\Big). $$
Symmetric $q$ (random-walk) ⇒ ratio collapses to $p(\theta')/p(\theta)$ (Metropolis). Only needs the **unnormalized** posterior — the normalizing constant cancels (this is the whole point; tie to Ch 01 evidence). The accept rule is precisely engineered so $T$ satisfies detailed balance for $p$.

**Gibbs.** Special MH with proposal = exact full conditional $p(\theta_i \mid \theta_{-i}, y)$; acceptance is always 1. Needs tractable conditionals (conjugacy). Fails badly under strong posterior correlation (zig-zags one axis at a time). This is *why* BUGS/JAGS struggled and why HMC won.

**Why random-walk MCMC fails in high dimensions — the typical set (Betancourt's framing — must teach this carefully).**
- The posterior mass is **not** where the density is highest. Density $p(\theta)$ is largest at the mode, but the **volume** of parameter space grows like $r^{d-1}$ as you move out. Expected value of any function is `density × volume`. In high $d$ these two fight, and the product concentrates on a thin shell — the **typical set** — away from the mode.
- Concentration of measure: as $d\to\infty$ the typical set becomes a thinner and thinner shell at near-constant radius; almost all probability lives there, the mode contributes negligible mass.
- Random-walk proposals are isotropic and blind to this geometry. Step too small → glacial diffusion along the shell (needs $O(d)$+ steps to traverse, ESS collapses). Step too big → proposals overshoot the thin typical set into the low-probability interior/exterior → rejected. The "Goldilocks" step shrinks with $d$; acceptance and efficiency degrade with dimension. Gibbs has the analogous problem under correlation.
- The fix is to **follow the geometry of the typical set** rather than guess. That requires the *gradient* of $\log p$, which points toward higher density and, combined with momentum, lets you glide *along* the typical set.

**Hamiltonian Monte Carlo.** Augment $\theta$ (position) with auxiliary momentum $r$, sampled $r\sim\mathcal N(0,M)$. Define potential and kinetic energy:
$$ U(\theta) = -\log p(\theta\mid y)\ (\text{up to const}),\qquad K(r) = \tfrac12 r^\top M^{-1} r, $$
$$ H(\theta,r) = U(\theta) + K(r). $$
The joint $\propto e^{-H} = p(\theta\mid y)\,\mathcal N(r\mid 0,M)$ — marginalizing out $r$ recovers the posterior. Simulate Hamilton's equations:
$$ \dot\theta = \partial H/\partial r = M^{-1}r,\qquad \dot r = -\partial H/\partial\theta = -\nabla U(\theta) = \nabla\log p(\theta\mid y). $$
Exact dynamics conserve $H$ and volume → would be accepted with prob 1. The gradient $\nabla\log p$ is what lets the trajectory **flow along the typical set** instead of diffusing across it; that's why gradients help.

**Leapfrog (Störmer–Verlet) integrator** — symplectic, time-reversible, volume-preserving, $O(\epsilon^2)$ error per step; this is why it's used over plain Euler. One step of size $\epsilon$:
$$ r_{t+\epsilon/2} = r_t + \tfrac{\epsilon}{2}\nabla\log p(\theta_t) $$
$$ \theta_{t+\epsilon} = \theta_t + \epsilon\,M^{-1} r_{t+\epsilon/2} $$
$$ r_{t+\epsilon} = r_{t+\epsilon/2} + \tfrac{\epsilon}{2}\nabla\log p(\theta_{t+\epsilon}) $$
Half-step momentum, full-step position, half-step momentum ("kick-drift-kick"). Because integration is approximate, $H$ drifts slightly; a Metropolis accept on $\exp(H_{\text{old}}-H_{\text{new}})$ corrects the discretization bias exactly. Momentum is **refreshed** (resampled from $\mathcal N(0,M)$) at the start of each iteration — this is what makes successive trajectories explore different directions and the whole thing ergodic.

**Divergences (define here, deep-dive in Ch 07).** When the true energy surface has high curvature (e.g. the neck of a funnel), a fixed $\epsilon$ leapfrog becomes unstable, the numerical $H$ blows up, $\big|H_{\text{new}}-H_{\text{old}}\big|$ exceeds a threshold → PyMC flags a **divergence** and stops that trajectory. Divergences are biased-inference alarms: the sampler is failing to explore a region (often the funnel neck). Raising `target_accept` shrinks $\epsilon$; reparameterizing (non-centered) reshapes the geometry. Some divergences are false positives, but a cluster of them in one region is real.

**NUTS — No-U-Turn Sampler (Hoffman & Gelman 2014, JMLR 15).** Plain HMC needs you to pick the number of leapfrog steps $L$ (trajectory length). Too short → random-walk-like; too long → wastes computation looping back ("U-turn"). NUTS builds the trajectory by **recursive doubling**: simulate forward and backward in time, doubling the number of leapfrog steps each iteration, building a balanced binary tree of states. Stop when any sub-trajectory starts to **double back on itself** — the no-U-turn criterion: for endpoints with positions $\theta^+,\theta^-$ and momenta $r^+,r^-$,
$$ (\theta^+ - \theta^-)\cdot r^- < 0 \quad\text{or}\quad (\theta^+ - \theta^-)\cdot r^+ < 0 . $$
i.e. continuing would reduce the distance between the ends. The next state is sampled (slice/multinomial) from the tree's states, preserving detailed balance. `max_treedepth` caps the doubling (default 10 → ≤1024 steps); hitting it is an *efficiency* warning (often a poorly-scaled metric or too-low step size), not a correctness error. Modern Stan/PyMC use the **multinomial** sampler from the trajectory rather than the original slice variant.

**Warmup / adaptation (two things tuned during `tune`):**
1. **Step size $\epsilon$ via dual averaging** (Nesterov primal-dual averaging; Hoffman & Gelman Alg. 6). $\epsilon$ is adapted so the average Metropolis acceptance approaches `target_accept` $\delta$ (default 0.8). Higher $\delta$ → smaller $\epsilon$ → shorter, more accurate steps → fewer divergences but slower (longer trajectories, deeper trees). Raise to 0.9/0.95/0.99 when fighting divergences. Diminishing returns + much higher cost past ~0.99.
2. **Mass matrix / metric $M$.** $M$ rescales/rotates the space so the kinetic energy "knows" the posterior covariance; ideally $M^{-1}\approx \text{Cov}(\theta\mid y)$, making the typical set look isotropic and trajectories efficient. `adapt_diag` learns a diagonal (per-parameter scales); `adapt_full` learns the full covariance (handles correlations, costs O(d²)). Adaptation happens in expanding windows during `tune`; tuning draws are discarded (`discard_tuned_samples=True`). If you don't give enough `tune`, $M$ and $\epsilon$ are mis-set → spurious divergences/low ESS; bump `tune` to 2000–3000 for hard models.

**Energy / BFMI (define here, deep-dive Ch 07).** `az.plot_energy` overlays the **marginal energy distribution** with the **energy-transition distribution** (successive $\Delta E$). If the transition distribution is much narrower than the marginal, the momentum resampling can't move energy fast enough → slow mixing in the tails. **BFMI** (Bayesian Fraction of Missing Information) quantifies this; **BFMI < 0.3 (some say 0.2)** flags a problem, usually cured by reparameterization.

## Common errors → causes → fixes

| Symptom / message | Cause | Fix |
|---|---|---|
| `There were N divergences after tuning` | Step size too large for high-curvature region (funnel neck); pathological geometry | Raise `target_accept` (0.9→0.95→0.99); **reparameterize non-centered**; tighten priors; rescale/standardize predictors |
| `The acceptance probability does not match the target` | Adaptation didn't converge; too few tuning steps | Increase `tune` (2000–3000); set `init="jitter+adapt_diag"` explicitly |
| `Maximum tree depth reached` / many `tree_depth==max_treedepth` | Trajectories long because metric poorly scaled or $\epsilon$ tiny | Improve mass matrix (`init="adapt_full"` if correlated), standardize, increase `tune`, only then raise `max_treedepth` |
| `The rhat statistic is larger than 1.01` | Chains haven't mixed / multimodality / non-convergence | More `tune`+`draws`; reparameterize; check model identifiability (defer detail to Ch 07) |
| `The effective sample size per chain is smaller than 100` | High autocorrelation / poor geometry | Reparameterize, raise `target_accept`, longer chains |
| Sampling extremely slow, low ESS, no divergences | Strong posterior correlations the diagonal metric can't capture | `init="jitter+adapt_full"` (dense metric), or reparameterize to decorrelate |
| `SamplingError: Initial evaluation ... results in -inf` | Bad initial point / impossible value under prior (e.g. obs outside support) | Set `initvals=`, fix priors/transforms, check `observed` data domain |
| `BlasError` / multiprocessing hangs on some OS | mp context / BLAS thread contention | `cores=1` to debug; set `mp_ctx="spawn"`; `blas_cores=1` |
| Discrete params + NUTS error "Cannot sample" | NUTS needs gradients; discrete vars unsupported directly | Marginalize the discrete var (Ch 13) or let PyMC use CompoundStep (`*GibbsMetropolis`) automatically |
| Metropolis "works" but ESS tiny in high-D | Random-walk diffusion / typical-set problem (the chapter's core lesson) | Use NUTS (gradient-based); this is the teachable moment |

## Datasets & load snippets

- **Eight Schools** (canonical for funnel/divergences/non-centering). Hardcode:
```python
J = 8
y     = np.array([28., 8., -3., 7., -1., 1., 18., 12.])
sigma = np.array([15., 10., 16., 11., 9., 11., 10., 18.])
```
- **2-D correlated Gaussian / banana / funnel** — best synthetic demos for "random-walk vs HMC". Neal's funnel (the HMC stress test):
```python
# v = log-scale, x_i | v ~ Normal(0, exp(v/2)); v ~ Normal(0,3)
with pm.Model() as funnel:
    v = pm.Normal("v", 0, 3)
    x = pm.Normal("x", 0, pm.math.exp(v/2), shape=9)
    idata = pm.sample(1000, tune=1000, target_accept=0.95, random_seed=8927)
```
- **Howell !Kung** (R2/Ch01 dataset) for a "real" first NUTS fit if a regression example is wanted: `pd.read_csv("https://raw.githubusercontent.com/rmcelreath/rethinking/master/data/Howell1.csv", sep=";")`.
- Encourage **synthetic-with-known-truth** to show step-size/dimension effects: generate `np.random.multivariate_normal` in d=2,10,100 and compare Metropolis ESS vs NUTS ESS to *demonstrate* the dimension scaling claim numerically.

## Curated resources (each: real URL + why + pointer)

- Betancourt, "A Conceptual Introduction to Hamiltonian Monte Carlo," arXiv:1701.02434 — **the** source for typical set, why RW fails, HMC geometry. https://arxiv.org/abs/1701.02434 (PDF https://arxiv.org/pdf/1701.02434). Read §2 (computing expectations / typical set), §3 (Hamiltonian dynamics), §4–5 (implementation, integrator).
- Neal, "MCMC using Hamiltonian dynamics," Handbook of MCMC (2011), arXiv:1206.1901 — definitive leapfrog + momentum + correctness derivation. https://arxiv.org/pdf/1206.1901 (also Gelman mirror https://sites.stat.columbia.edu/gelman/bayescomputation/Neal2011.pdf). §5.2 leapfrog, §5.3 the HMC algorithm.
- Hoffman & Gelman, "The No-U-Turn Sampler," JMLR 15 (2014) — NUTS + dual averaging; Algorithm 6 is the canonical pseudocode. https://jmlr.org/papers/v15/hoffman14a.html
- Betancourt, "Markov Chain Monte Carlo in Practice" (case study) — typical set, concentration of measure, transitions drifting to typical set, in plain language. https://betanalpha.github.io/assets/case_studies/markov_chain_monte_carlo.html
- Betancourt, "Markov Chain Monte Carlo Basics" (case study) — gentler companion. https://betanalpha.github.io/assets/case_studies/markov_chain_monte_carlo_basics.html
- PyMC docs, `pymc.sample` API — verify every knob/default. https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.sample.html
- PyMC docs, `pymc.init_nuts` — exact `init` strings + metric adaptation semantics. https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.init_nuts.html
- PyMC docs, `pymc.NUTS` step method — `target_accept`, `max_treedepth`, `step_scale` args. https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.NUTS.html
- PyMC example gallery, "Diagnosing Biased Inference with Divergences" — divergence access idioms, non-centered eight-schools, target_accept/tune sweeps. https://www.pymc.io/projects/examples/en/latest/diagnostics_and_criticism/Diagnosing_biased_Inference_with_Divergences.html
- PyMC example gallery, "Sampler Statistics" — what's in `sample_stats` (energy, tree_depth, step_size). https://www.pymc.io/projects/examples/en/latest/diagnostics_and_criticism/sampler-stats.html
- PyMC example gallery, "Compound Steps in Sampling" — how PyMC mixes NUTS + discrete steppers. https://www.pymc.io/projects/examples/en/latest/samplers/sampling_compound_step.html
- PyMC example gallery, DEMetropolis(Z) comparison — gradient-free MCMC contrast (when to NOT use NUTS). https://www.pymc.io/projects/examples/en/latest/samplers/DEMetropolisZ_EfficiencyComparison.html
- Stan Reference Manual, HMC/NUTS chapter — independent authoritative description of metric, step size, treedepth. https://mc-stan.org/docs/reference-manual/mcmc.html
- Martin, Kumar, Lao, *Bayesian Modeling and Computation in Python* — free online, Ch 2/Ch 11 cover inference engines + HMC/NUTS intuition with PyMC/ArviZ. https://bayesiancomputationbook.com
- ArviZ `plot_energy` docs — BFMI/energy plot reading. https://python.arviz.org/en/stable/api/generated/arviz.plot_energy.html
- (Optional, intuition) Chi Feng's interactive MCMC demo — visualize RW-Metropolis vs HMC vs NUTS on funnel/banana. https://chi-feng.github.io/mcmc-demo/

## Uncertainties / things the writer should double-check

- **`target_accept` default**: top-level `pm.sample` forwards to NUTS whose default is **0.8**. The course bible's house style shows `target_accept=0.9` as the default *teaching* call — that's a deliberate course convention, not the library default. Make the distinction explicit so the chapter isn't internally contradictory with Ch 07.
- **`tune` default**: the live docstring says "defaults to 1000" even though the signature shows `tune=None` (None resolves to 1000 internally). State 1000.
- **ArviZ API drift on divergence plotting**: newer ArviZ moved some args under `visuals={...}` (e.g. `az.plot_pair(..., visuals={"divergence": True})`) while stable still accepts `divergences=True`. Verify against the installed ArviZ version; show the stable spelling and add a `# newer ArviZ: visuals={"divergence": True}` comment.
- **No `pm.Gibbs` class**: confirm there is no standalone Gibbs stepper; Gibbs is taught as concept + realized via `*GibbsMetropolis` discrete steppers / custom conjugate `pm.step_methods` (see "sampling_conjugate_step" gallery). Don't invent `pm.Gibbs`.
- **No-U-turn criterion exact form**: the dot-product condition above is the standard statement (Hoffman & Gelman eq. for the termination criterion); verify the precise sign/indexing against §3 of the JMLR paper before quoting it as an equation — there are slightly different but equivalent statements (generalized U-turn uses the metric $M^{-1}$).
- **Slice vs multinomial NUTS**: the 2014 paper uses a slice sampler over the trajectory; current Stan/PyMC use multinomial sampling. Say "modern implementations use multinomial sampling" rather than implying the original slice variant is what PyMC runs.
- **BFMI threshold**: literature quotes both <0.2 and <0.3 as the warning line; bible standardizes on **<0.3** (§10). Use 0.3 and note 0.2 is also seen.
- **`HamiltonianMC` exists but is rarely used** — confirm `pm.HamiltonianMC` is still exported in the installed v5 (it is in 5.10.x docs); recommend NUTS for all teaching examples.
- Could not WebFetch the full betanalpha MCMC-in-practice page (exceeded fetch size limit); the typical-set/concentration points above are from the search abstract + the arXiv 1701.02434 framing. The writer should skim the case study directly for the exact "drift into the typical set" wording.
