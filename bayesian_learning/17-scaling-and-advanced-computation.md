# Chapter 17 — Scaling & Advanced Computation

### What to do when the default sampler is too slow — alternative NUTS backends, GPUs, within-chain parallelism, minibatch VI, and the performance traps that quietly cost you hours

> *A note from me to you.*
>
> *Up to now we have treated `pm.sample()` as a black box that always finishes. For most of the models in this course it does, and quickly — eight schools fits in seconds, a hierarchical radon model in a minute or two. But the day will come, and it will come sooner than you think, when you press enter on a model you are proud of and the progress bar tells you it will be done in **nine hours**. Or it tells you forty minutes per chain and you have a deadline in twenty. This is the chapter that keeps that day from derailing you.*
>
> *Here is the mistake I want to inoculate you against: when sampling is slow, the junior reaches for the wrong knob. They crank `target_accept` to 0.99 "to be safe" (and triple the runtime), or they throw the model at a GPU that makes it **slower**, or they buy a bigger machine when the real problem is a Python `for` loop inside the model. Speed in Bayesian computation is not one lever; it's a small decision tree, and most of the wins come from understanding **where your time actually goes** — compile time versus sampling time, gradient evaluation versus tree-building, Python overhead versus genuine arithmetic. Measure first, then choose the tool. A five-minute profile saves a five-hour wait.*
>
> *We'll go through the whole toolkit in the order you should actually consider it. First the free win that asks for one keyword: swapping the NUTS backend (NumPyro, nutpie, BlackJAX) under PyMC's `nuts_sampler=` argument. Then when a GPU helps and when it's a trap. Then the structural wins — vectorizing away loops, marginalizing out latents, sparse GP approximations — that change the model itself so the sampler has an easier job. Then the large-data tools: minibatch ADVI, and Stan's within-chain `reduce_sum` parallelism contrasted with PyMC's cross-chain multiprocessing. We close with a decision guide you can pin above your desk. Nothing here changes the **statistics** — the posterior is the same target it always was. This chapter is purely about reaching it faster, and about not fooling yourself into "faster" that's actually broken.*

---

## What you'll be able to do after this chapter

- **Profile a PyMC model** to separate compile time from sampling time, and read `idata.sample_stats` to find where the sampler is spending its leapfrog steps — so you optimize the thing that's actually slow.
- **Swap the NUTS backend** with a single keyword — `pm.sample(nuts_sampler="numpyro" | "nutpie" | "blackjax")` — know what each backend is (JAX vmap/JIT, fast Rust NUTS, composable JAX primitives), and pass backend-specific options through `nuts_sampler_kwargs`.
- **Decide whether a GPU will help**, understanding *why* it's about big vectorized models and many chains, not about small models — and how to run chains in parallel on one GPU with `chain_method="vectorized"`.
- **Use Stan's `reduce_sum`** for within-chain parallelism on a summable log-likelihood, and explain how it differs from PyMC's cross-chain `cores` multiprocessing.
- **Fit models on data too large for full-batch gradients** with minibatch ADVI (`pm.Minibatch` + `total_size`), and state honestly what that approximation costs you.
- **Speed the sampler up by changing the model**: vectorize away Python loops, marginalize discretes and latents (recap of Chapters 13 and 16), and use HSGP / sparse approximations for GPs (recap of Chapter 12).
- **Re-predict without recompiling** using `pm.Data` + `pm.set_data`, and recognize the common performance traps — unvectorized models, a dense mass matrix where a diagonal suffices, `target_accept` set needlessly high, silent recompilation.
- **Choose a sampler/backend deliberately** from a decision guide keyed to model size, data size, hardware, and whether you have gradients.

This chapter is the operational sequel to the inference chapters. It assumes you understand *how* NUTS works from **Chapter 05 (Inference 2: MCMC from Metropolis to NUTS)** — the leapfrog integrator, the mass matrix / metric, dual-averaging step-size adaptation, what `target_accept` and `max_treedepth` actually do — and that you read every fit through the diagnostics of **Chapter 07 (MCMC Diagnostics & Debugging)** ($\hat R \le 1.01$, ESS$_\text{bulk}$ and ESS$_\text{tail} \gtrsim 400$, zero divergences, BFMI $> 0.3$). We lean on the marginalization ideas from **Chapter 13 (Mixture Models & Latent Variables)** and **Chapter 16 (Missing Data the Bayesian Way)**, the HSGP/sparse approximations from **Chapter 12 (Gaussian Processes)**, and the variational machinery (ELBO, mean-field vs full-rank, *where VI lies to you*) from **Chapter 06 (Inference 3: Variational Inference)**.

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

## 1. First, find out where the time actually goes

Before you change a single thing, you need to know what "slow" means *for your model*, because the cure is completely different depending on the answer. There are four places a Bayesian fit spends time, and they respond to different fixes:

1. **Compile time** — PyMC builds a computational graph (PyTensor) for your log-density and its gradient, then a backend (C, Numba, or JAX) compiles it to fast machine code. This happens **once** per model, before any sampling. For a big model this can be seconds to a minute. If you re-run a notebook cell that rebuilds the model, you pay it again.
2. **Gradient evaluation cost** — every leapfrog step needs one evaluation of $\nabla \log p(\theta \mid y)$. The cost of that single evaluation scales with the size of your data and the arithmetic in your model. This is the per-step cost.
3. **Number of leapfrog steps** — NUTS takes a variable number of steps per draw (it builds a trajectory by doubling, up to $2^{\texttt{max\_treedepth}}$). A badly-scaled posterior, a funnel, or a high `target_accept` all push this number up. This is the steps-per-draw count.
4. **Number of draws × chains** — you asked for this much, multiplied across chains.

The total *compute per chain* is roughly **(gradient cost) × (steps per draw) × (draws)**, plus a one-time compile. The **wall-clock** is that compute divided by how many chains run in parallel — with across-chain multiprocessing (§5), four chains on four cores finish in roughly one chain's wall-time, so chains multiply *compute* but not *wall-clock* when you have the cores. The single most useful habit in this whole chapter is to *attribute your slowness to one of these four lines* before reaching for a tool, because:

- If **compile** dominates (you're iterating on a small model and recompiling constantly), a faster *sampler* won't help — you need to stop recompiling, or accept a one-time JAX/Numba compile that pays off over a long run.
- If **gradient cost** dominates (big data, heavy arithmetic), that's where GPUs, vectorization, and minibatching live.
- If **steps per draw** dominates (deep trees, lots of divergences), the fix is *geometry* — reparameterize, fix the mass matrix, standardize — not raw horsepower.
- If you just asked for **too many draws**, well.

> 💡 **Intuition — slowness has a shape.** "My model is slow" is like "my car won't start." Useless until you localize it. A model that's slow because each gradient touches a million data points is a *different problem* from a model that's slow because the sampler takes 1000 leapfrog steps per draw fighting a funnel. The first wants a GPU; the second wants a reparameterization, and a GPU would do nothing. Diagnose before you medicate.

### 1.1 Profiling compile vs sample time

The crudest, most honest profiler is a wall clock around the two phases. Let me show you the pattern explicitly, because separating these two numbers is the whole game:

```python
import time

# --- a small model to demonstrate the timing pattern ---
X = rng.normal(size=(2000, 5))
true_beta = np.array([1.0, -2.0, 0.5, 0.0, 3.0])
y = X @ true_beta + rng.normal(scale=1.0, size=2000)

coords = {"pred": [f"x{i}" for i in range(5)], "obs": np.arange(len(y))}

t0 = time.perf_counter()
with pm.Model(coords=coords) as model:
    beta = pm.Normal("beta", 0, 2, dims="pred")
    sigma = pm.HalfNormal("sigma", 1)
    mu = pm.Deterministic("mu", pm.math.dot(pm.Data("X", X, dims=("obs", "pred")), beta), dims="obs")
    pm.Normal("y", mu=mu, sigma=sigma, observed=y, dims="obs")
t_build = time.perf_counter() - t0   # graph construction (cheap)

t0 = time.perf_counter()
with model:
    idata = pm.sample(1000, tune=1000, chains=4, target_accept=0.9,
                      random_seed=RANDOM_SEED)
t_sample = time.perf_counter() - t0   # compile + warmup + sampling, all together

print(f"graph build: {t_build:.2f}s   sample (incl. compile+warmup): {t_sample:.2f}s")
```

That `t_sample` number bundles compile, warmup, and sampling together — useful as a top-line, but to split compile out you want PyTensor's own profiler, which times the *function* PyMC compiles. PyMC exposes this through `model.profile`:

```python
# Profile the compiled logp and its gradient — counts the real arithmetic, not Python overhead.
# model.profile(outs) compiles the graph for `outs`, runs it (n=1000 iterations by default),
# and returns a PyTensor ProfileStats object you can .summary().
prof_logp = model.profile(model.logp())     # profile the log-density graph
prof_logp.summary()                          # prints per-Op time: which operations dominate

# The gradient is what NUTS actually evaluates at EVERY leapfrog step, so it's the per-step
# bottleneck — profile it too (this is usually the number that matters for sampling speed):
prof_grad = model.profile(model.dlogp())    # profile the GRADIENT graph
prof_grad.summary()
# verify in current PyMC docs: model.profile signature has shifted across versions
```

What `prof_grad.summary()` shows you is a table of PyTensor *Ops* (matrix multiply, the Normal log-density, the gradient pieces) sorted by total time. If one `Dot` or one `Elemwise` is eating 90% of the gradient evaluation, *that* is the thing to vectorize, sparsify, or push to a GPU. If instead you see hundreds of tiny Ops each taking a sliver — that's the signature of an **unvectorized model** (a Python loop that built hundreds of scalar nodes), and the fix is §6, not a new backend.

> 🩺 **Diagnostic — read the steps, not just the seconds.** After any fit, look at `idata.sample_stats`. The keys you care about for performance are `n_steps` (leapfrog steps per draw), `tree_depth` (the doubling depth — if it's pinned at 10 = `max_treedepth`, NUTS is hitting its cap and burning the maximum 1024 steps every draw), and `step_size` (a tiny adapted step size means a cramped geometry forcing many small steps). High `n_steps` with no divergences usually means a poorly-scaled mass matrix or genuine correlation — a *geometry* problem, not a horsepower problem.

```python
ss = idata.sample_stats
print("mean leapfrog steps/draw:", float(ss["n_steps"].mean()))
print("max tree_depth hit:", int(ss["tree_depth"].max()), "(cap is max_treedepth, default 10)")
print("adapted step_size (per chain):", ss["step_size"].mean(dim="draw").values)
print("divergences:", int(ss["diverging"].sum()))
```

> 🔧 **In practice — the cheapest speedup is usually free.** Before any of the heavy machinery in this chapter, ask: am I recompiling needlessly (re-running the model-definition cell)? Did I set `target_accept` higher than I need? Am I sampling 4000 draws when 1000 gives ESS in the thousands already? Am I running on a laptop with `cores=1` by accident? Fix those, *then* reach for a backend. I have seen a "switch to GPU" project that was really a "stop rebuilding the model in a loop" project.

---

## 2. The one-keyword win: alternative NUTS backends

Here is the most important practical fact in this chapter, and the first thing to try when sampling is slow: **PyMC can hand the exact same model to a different NUTS implementation, and you select it with one keyword.**

```python
idata = pm.sample(1000, tune=1000, chains=4, target_accept=0.9,
                  nuts_sampler="numpyro",        # <-- the whole trick
                  random_seed=RANDOM_SEED)
```

`nuts_sampler` takes one of four values: `"pymc"` (the default), `"nutpie"`, `"numpyro"`, or `"blackjax"`. **The model code does not change at all.** You write your `pm.Model` block exactly as always; PyMC translates the log-density graph into the form the chosen backend wants and runs *that backend's* NUTS. The returned object is still an `az.InferenceData`, your diagnostics in **Chapter 07** are identical, and the posterior is the same target. What changes is the engine doing the leapfrog integration and tree-building — and engines differ enormously in speed.

Why does swapping the engine help at all, if it's "the same NUTS"? Because the four backends differ in (a) what they compile to (C, Numba, or JAX/XLA), (b) how aggressively they fuse and vectorize the arithmetic, (c) their warmup schedule and mass-matrix adaptation, and (d) whether they can use a GPU. For a model with a lot of vectorized arithmetic, a JAX backend that JIT-compiles the whole gradient into one fused XLA kernel can be several times faster than the default per-Op execution. For a moderate CPU model, nutpie's Rust NUTS with a particularly good warmup often wins on both compile time and samples-per-second.

Let me introduce each one properly, because choosing among them is a real skill.

### 2.1 The default — `"pymc"`

The native sampler. NUTS implemented in Python orchestration over a log-density compiled by PyTensor to **C or Numba**. Its great virtue is **compatibility**: it is the only backend that can sample a model containing **discrete random variables** (it builds a `CompoundStep`, assigning NUTS to the continuous parameters and a Metropolis/Gibbs variant to the discrete ones — exactly the coal-mining switchpoint mechanism from **Chapter 15**, building on the marginalize-the-discrete ideas introduced in **Chapter 13**). The JAX and Rust backends below require a **fully continuous** model graph; hand them a discrete RV and they cannot run. So `"pymc"` is your fallback whenever the model has a discrete latent you haven't marginalized out, and it's the most thoroughly battle-tested path.

```python
# the default — equivalent to not passing nuts_sampler at all
idata = pm.sample(1000, tune=1000, chains=4, target_accept=0.9,
                  nuts_sampler="pymc", random_seed=RANDOM_SEED)
```

### 2.2 nutpie — fast Rust NUTS, the best CPU default

**nutpie** is a NUTS implementation written in **Rust** with a notably good warmup/adaptation schedule. It compiles your PyMC model's log-density (via Numba **or** JAX) and then runs the actual sampler — tree-building, step-size and mass-matrix adaptation — in tight, multi-threaded Rust. In practice nutpie is frequently the **fastest CPU option** and has a **fast compile**, which makes it lovely for iterative work where you fit, look, tweak, refit. It also implements a slightly more sophisticated mass-matrix adaptation than the default, which can mean fewer tuning steps to reach a good metric.

```python
idata = pm.sample(1000, tune=1000, chains=4, target_accept=0.9,
                  nuts_sampler="nutpie", random_seed=RANDOM_SEED)

# choose nutpie's compile backend explicitly; "jax" enables GPU, "numba" is CPU-only
idata = pm.sample(nuts_sampler="nutpie",
                  nuts_sampler_kwargs={"backend": "jax", "gradient_backend": "jax"},
                  random_seed=RANDOM_SEED)
```

Install it with `pip install nutpie` (and `numba` for the CPU backend, or `jax` for the JAX backend). Its Numba backend is CPU-only but is often the single fastest thing you can run on a laptop; its JAX backend unlocks the GPU path of §3.

> 🔧 **In practice — try nutpie first.** When a model that has no discrete variables samples too slowly with the default, my first move is literally to add `nuts_sampler="nutpie"` and re-run. It's a thirty-second experiment that frequently halves the wall-clock for free, with a posterior identical to the default's. If that's not enough, *then* think about GPUs and minibatching.

### 2.3 NumPyro — JAX, vmap/JIT, and the road to the GPU

**NumPyro** is a probabilistic-programming library built on **JAX**. When you pass `nuts_sampler="numpyro"`, PyMC compiles your log-density to JAX and runs NumPyro's NUTS, which is itself JAX. The two superpowers JAX brings are **JIT compilation** (the whole gradient is fused by XLA into one optimized kernel, eliminating Python-level per-Op overhead) and **`vmap` vectorization** (it can run all your chains as a single vectorized computation rather than separate processes). The same JAX code runs on CPU **or GPU/TPU** with no change to your model — which is exactly why NumPyro is the backend you reach for when you want a GPU (§3).

```python
idata = pm.sample(1000, tune=1000, chains=4, target_accept=0.9,
                  nuts_sampler="numpyro", random_seed=RANDOM_SEED)

# pass NumPyro-specific options through nuts_sampler_kwargs:
idata = pm.sample(nuts_sampler="numpyro",
                  nuts_sampler_kwargs={
                      "chain_method": "vectorized",     # run chains via vmap (one device, parallel)
                      "postprocessing_backend": "cpu",  # move post-processing off the GPU
                  },
                  random_seed=RANDOM_SEED)
```

Install with `pip install numpyro` (CPU) plus the appropriate `jax[cuda12]` build for GPU. The `chain_method` option matters: `"parallel"` runs chains on separate devices, `"vectorized"` runs them all together via `vmap` (ideal for a single GPU), and `"sequential"` runs them one after another (low memory).

The `postprocessing_backend="cpu"` option needs a word too, because it's easy to hit. After sampling, NumPyro builds the `InferenceData` and computes the `sample_stats` (and any deterministics) — and by default it does that *on the same device it sampled on*. On a memory-tight GPU, that post-processing step can **OOM** even when the sampling itself fit, because it materializes the full draws×chains arrays on-device. Setting `postprocessing_backend="cpu"` moves that build-and-summarize step to host RAM, leaving the GPU memory for the sampling that actually needs it. It's the standard fix for "sampling finished but then the run died with an out-of-memory error."

### 2.4 BlackJAX — composable JAX inference primitives

**BlackJAX** is a library of **composable inference building blocks** in JAX — NUTS, HMC, SMC, Pathfinder, window adaptation, and more, exposed as small functional pieces you can assemble. Through PyMC's `nuts_sampler="blackjax"` you get its NUTS on your model, also JAX-backed and GPU-capable. Its niche is **research and flexibility**: if you want to swap in a custom adaptation scheme, a different kinetic energy, or compose NUTS inside a tempering/SMC loop, BlackJAX is built for that. For everyday "just sample my model faster," nutpie or NumPyro are the more common picks; reach for BlackJAX when you want to *modify the algorithm*, not just run it.

```python
idata = pm.sample(1000, tune=1000, chains=4, target_accept=0.9,
                  nuts_sampler="blackjax", random_seed=RANDOM_SEED)
```

Install with `pip install blackjax` (plus JAX). There are also **low-level direct entry points** that bypass `pm.sample` entirely, if you want to drive the JAX samplers yourself:

```python
import pymc.sampling.jax  # verify in current PyMC docs (module path is pymc.sampling.jax)

with model:
    idata = pymc.sampling.jax.sample_numpyro_nuts(1000, tune=1000, chains=4,
                                                  target_accept=0.9, random_seed=RANDOM_SEED)
    # and the BlackJAX twin:
    # idata = pymc.sampling.jax.sample_blackjax_nuts(1000, tune=1000, chains=4)
```

### 2.5 How backend options are passed — `nuts_sampler_kwargs`

This is the single API detail to get right. Top-level arguments you already know — `draws`, `tune`, `chains`, `target_accept`, `random_seed` — are passed to `pm.sample` directly and **forwarded to whichever backend you chose**. Anything *specific to a backend* goes in the dictionary `nuts_sampler_kwargs`:

```python
idata = pm.sample(
    1000, tune=1000, chains=4, target_accept=0.9,   # generic, forwarded to the backend
    nuts_sampler="numpyro",
    nuts_sampler_kwargs={                            # backend-specific knobs live here
        "chain_method": "vectorized",
    },
    random_seed=RANDOM_SEED,
)
```

| Backend | Compiles to | Key `nuts_sampler_kwargs` | GPU? | Discrete RVs? |
|---|---|---|---|---|
| `"pymc"` | C / Numba | (uses top-level NUTS kwargs) | No | **Yes** (CompoundStep) |
| `"nutpie"` | Numba **or** JAX | `backend`, `gradient_backend` | Via JAX backend | No |
| `"numpyro"` | JAX | `chain_method`, `postprocessing_backend` | **Yes** | No |
| `"blackjax"` | JAX | `chain_method`, adaptation options | **Yes** | No |

> ⚠️ **Pitfall — the JAX/Rust backends cannot sample discrete parameters.** If your model has a `pm.DiscreteUniform` switchpoint, a `pm.Categorical` mixture index you haven't marginalized, or any other discrete RV that needs a Metropolis/Gibbs step, `nuts_sampler="numpyro"` (and nutpie, and blackjax) will fail — they are pure-NUTS engines and NUTS needs continuous gradients. The fix is either stay on `"pymc"` (which builds a CompoundStep) or **marginalize the discrete out** (§5) so the whole graph is continuous, at which point any backend will take it.

---

## 3. GPUs: when they help, when they hurt

A GPU feels like it should make everything faster. It does not, and understanding *why* will save you from a classic disappointment: porting a small hierarchical model to a GPU and watching it run **slower** than your laptop.

A GPU is thousands of small arithmetic cores. It is spectacular at doing the **same operation across a huge array in parallel** — a giant matrix multiply, an elementwise log-density over a million data points. It is *bad* at small, sequential, branchy work, and it pays a fixed cost to move data on and off the device and to launch each kernel. MCMC is a fundamentally sequential algorithm — each leapfrog step depends on the previous one — so the GPU cannot parallelize the *time* dimension of a single chain at all. What it *can* parallelize is (a) the arithmetic **within a single gradient evaluation**, and (b) **many chains at once** via `vmap`.

That gives a clean rule:

> 💡 **Intuition — the GPU helps when each gradient is big.** A GPU pays off when one evaluation of $\nabla \log p$ involves a lot of parallel arithmetic — big data ($N$ large), big design matrices, big GP covariance operations, wide neural-net-like structures — and/or when you run many chains that can be vectorized together. It does **not** help a small model (eight schools, a 5-parameter regression on 2000 rows): the per-step arithmetic is so small that kernel-launch and data-transfer overhead dominate, and the sequential nature of MCMC means all those idle cores sit waiting. For small models the GPU is *slower*. The break-even is "is each gradient evaluation large enough to saturate the device?"

Concretely, you reach the GPU through a JAX backend. Install the CUDA build of JAX (`pip install -U "jax[cuda12]"`, following JAX's current platform instructions — this evolves, so check the JAX install docs), and then:

```python
# Run all chains in parallel on a single GPU via vmap. The model code is unchanged.
idata = pm.sample(
    1000, tune=1000, chains=4, target_accept=0.9,
    nuts_sampler="numpyro",
    nuts_sampler_kwargs={
        "chain_method": "vectorized",     # vmap the chains onto one GPU
        "postprocessing_backend": "cpu",  # build InferenceData on CPU to save GPU memory
    },
    random_seed=RANDOM_SEED,
)
# verify in current JAX/NumPyro docs: jax[cuda12] install string and chain_method values evolve
```

`chain_method="vectorized"` is the magic for a single GPU: instead of four separate processes, NumPyro runs all four chains as one vectorized computation, so they share the device efficiently. (`"parallel"` wants multiple devices; `"vectorized"` wants one device, many chains — usually what you have.)

> 🔧 **In practice — the GPU checklist.** Use a GPU when: (1) your model has **large per-gradient arithmetic** — think $N$ in the hundreds of thousands, large GP kernels, big matrix ops; (2) you can express it in a **fully continuous, vectorized** JAX-compilable graph; (3) you want **many chains** run together. Skip the GPU when: the model is small (sub-second per chain on CPU already), the bottleneck is **compile time** (XLA compile on GPU can be *slower to start*), or the model has discrete RVs. And always benchmark — `time` the CPU run and the GPU run on *your* model. The GPU is a tool for a specific shape of problem, not a universal "go faster" button.

> ⚠️ **Pitfall — XLA compile time can dominate a short run.** The first JAX/GPU sample includes an XLA compilation that can take tens of seconds. If your *total* sampling is only a minute, that compile is a huge tax and the GPU "loses." JAX backends shine on **long** runs where the compile is amortized over many fast draws. Don't judge a backend by a 200-draw smoke test — the compile-to-sampling ratio is wildly different from a production run.

---

## 4. A worked end-to-end example: the same model, four backends, on a real dataset

Theory is cheap; let me put a real, reproducible model in front of you and actually race the backends on it. We'll use the **radon** dataset — the multilevel-modeling flagship from **Chapter 10** — because it's a genuine hierarchical model (varying intercepts by county) that is big enough to show backend differences but small enough to run on a laptop. This is the same data and the same partial-pooling structure you already understand; the *emphasis* here is on *how fast we can fit it* rather than *what it means* — but we still run the full course workflow (prior predictive → sample → diagnostics → posterior predictive), because speed is never an excuse to drop the checks.

### 4.1 Load and prepare the data

```python
# Radon (Minnesota), the canonical partial-pooling dataset (see Chapter 10).
srrs2 = pd.read_csv(pm.get_data("srrs2.dat"))
srrs2.columns = srrs2.columns.map(str.strip)
srrs_mn = srrs2[srrs2.state == "MN"].copy()

# log radon (+0.1 to avoid log(0)); county as a categorical grouping
srrs_mn["log_radon"] = np.log(srrs_mn["activity"].values + 0.1)
srrs_mn["county"] = srrs_mn["county"].str.strip()
county_idx, counties = pd.factorize(srrs_mn["county"])
floor = srrs_mn["floor"].values.astype(float)         # 0 = basement, 1 = first floor
log_radon = srrs_mn["log_radon"].values

coords = {"county": counties, "obs": np.arange(len(log_radon))}
print(len(log_radon), "houses across", len(counties), "counties")
```

### 4.2 The non-centered hierarchical model

We write the varying-intercept model in the **non-centered parameterization** by default — the funnel-curing reflex from **Chapter 07 / 10**. This matters doubly here: a centered version would force NUTS into a funnel, inflate `n_steps`, throw divergences, and make *every* backend slow. Good geometry is the prerequisite for any backend to be fast.

```python
with pm.Model(coords=coords) as radon_model:
    # data containers — pm.Data so we can re-predict later without recompiling (§9)
    floor_data = pm.Data("floor", floor, dims="obs")
    county_data = pm.Data("county_idx", county_idx, dims="obs")

    # population-level (hyper) parameters
    mu_a = pm.Normal("mu_a", 0.0, 1.0)          # mean county intercept
    sigma_a = pm.HalfNormal("sigma_a", 1.0)     # spread of county intercepts
    b = pm.Normal("b", 0.0, 1.0)                # floor (basement vs first) effect

    # NON-CENTERED varying intercepts: standardized offset, then rescale
    a_offset = pm.Normal("a_offset", 0.0, 1.0, dims="county")
    a = pm.Deterministic("a", mu_a + a_offset * sigma_a, dims="county")

    sigma_y = pm.HalfNormal("sigma_y", 1.0)
    mu = a[county_data] + b * floor_data
    pm.Normal("y", mu=mu, sigma=sigma_y, observed=log_radon, dims="obs")
```

> 💡 **Intuition — why this is the *right* model to benchmark.** A hierarchical model with hundreds of varying intercepts is exactly the regime where backend choice starts to matter: the gradient touches every county offset, so there's real vectorizable arithmetic, and the model is fully continuous (no discrete RVs), so *all four* backends can run it. It's the sweet spot for an apples-to-apples race.

Even though this chapter is about *speed*, we don't skip the workflow. Before sampling, a quick **prior predictive check** confirms the priors generate plausible log-radon values rather than nonsense — a five-second sanity gate the course never bypasses (Chapter 04):

```python
with radon_model:
    prior_pred = pm.sample_prior_predictive(draws=200, random_seed=RANDOM_SEED)
# log-radon in Minnesota homes realistically sits roughly in [-2, 4]; check the prior covers it
y_prior = prior_pred.prior_predictive["y"].values
print("prior predictive log-radon range:", float(y_prior.min()), "to", float(y_prior.max()))
```

The prior draws should span a sensible log-radon range (very roughly $[-2, 4]$) without exploding to $\pm 50$ — confirming the unit-scale `Normal`/`HalfNormal` priors are reasonable for standardized-ish data. If the range were wildly off, we'd fix the priors *before* spending any time benchmarking samplers.

### 4.3 Race the backends

```python
import time

def timed_sample(sampler, **kw):
    t0 = time.perf_counter()
    with radon_model:
        idata = pm.sample(1000, tune=1000, chains=4, target_accept=0.9,
                          nuts_sampler=sampler, random_seed=RANDOM_SEED,
                          progressbar=False, **kw)
    dt = time.perf_counter() - t0
    # ESS/sec is the metric that actually matters: useful samples per wall-second
    ess = az.ess(idata, var_names=["mu_a", "sigma_a", "b", "sigma_y"])
    min_ess = float(min(ess[v].values.min() for v in ess.data_vars))
    return idata, dt, min_ess, min_ess / dt

for name in ["pymc", "nutpie", "numpyro"]:    # add "blackjax" if installed
    try:
        idata, dt, min_ess, eff = timed_sample(name)
        print(f"{name:9s}  wall={dt:6.1f}s   min ESS={min_ess:7.0f}   ESS/sec={eff:6.1f}")
    except Exception as e:
        print(f"{name:9s}  (skipped: {type(e).__name__})")
```

> 🩺 **Diagnostic — compare on ESS/second, never on wall-clock alone.** The number that matters is **effective samples per wall-second** (the last column), not raw runtime. A backend that finishes in half the time but produces a third of the effective sample size is *worse*. ESS-per-second normalizes for the fact that different samplers, with different warmups and metrics, produce differently-autocorrelated draws. This is the honest benchmark; "it ran in 12 seconds" is not, because you don't know how *useful* those 12 seconds of draws are.

On a typical laptop, for a model this size, you'll usually see nutpie and the default `"pymc"` neck-and-neck and fast, with NumPyro paying a noticeable XLA **compile tax** that makes it *lose* on this small a model despite fast per-draw sampling — exactly the §3 lesson in miniature. Scale the dataset up 100× (or add 50 predictors) and the ranking flips toward the JAX backends. **The right backend depends on the size of the problem, which is why §10 is a decision tree and not a single recommendation.**

### 4.4 Verify the posterior is backend-invariant

The crucial sanity check: a faster backend must give the **same answer**. Confirm it with diagnostics and a side-by-side summary.

```python
# whichever idata you keep — confirm convergence is clean
az.summary(idata, var_names=["mu_a", "sigma_a", "b", "sigma_y"],
           kind="diagnostics")     # look at r_hat (<=1.01) and ess_bulk/ess_tail (>=400)
print("divergences:", int(idata.sample_stats["diverging"].sum()))
```

If two backends give materially different posteriors for the same model, that's a **bug or a non-convergence**, not a feature — investigate (usually one of them didn't actually converge, or a discrete RV silently broke a JAX run). When both converge, $\hat R \le 1.01$ and zero divergences, the posteriors will agree to Monte Carlo error. The backend is a *speed* choice; it is never a *statistics* choice.

> 🔧 **In practice — keep the fastest *converged* backend.** Pick the backend with the best ESS/second **among those that pass the Chapter 07 gates**. A backend that's twice as fast but throws divergences hasn't won; it's just wrong faster. Convergence is the entry ticket; speed is the tiebreaker.

Diagnostics tell you the *sampler* worked; a **posterior predictive check** tells you the *model* fits the data — and the course's workflow (Chapter 04) requires it even when speed is the headline. One terse block closes the loop:

```python
# posterior predictive check: does the fitted model reproduce the observed log-radon distribution?
with radon_model:
    pm.sample_posterior_predictive(idata, extend_inferencedata=True, random_seed=RANDOM_SEED)

az.plot_ppc(idata)   # observed log-radon density vs replicated draws from the posterior
```

The replicated densities should overlay the observed log-radon histogram closely — confirming the varying-intercept-plus-floor structure captures the data's location and spread. (If the observed distribution were bimodal or heavier-tailed than the replications, that would point to a missing predictor or the wrong likelihood — a *model* fix, not a *speed* one, and exactly the kind of thing a fast sampler will not save you from.)

---

## 5. Two kinds of parallelism: across chains vs within a chain

There are two completely different axes on which you can parallelize an MCMC fit, and conflating them is a common source of confusion. Get the distinction clear and the rest is easy.

**Across-chain parallelism (PyMC's `cores`).** You run 4 independent chains. They share nothing during sampling — each is its own Markov chain, started from a jittered point, exploring the posterior on its own. PyMC (and Stan, by default) runs them in **separate OS processes**, one per CPU core. This is *embarrassingly parallel*: 4 chains on 4 cores finish in roughly the wall-clock of 1 chain. You control it with `cores=` (and `chains=`):

```python
idata = pm.sample(1000, tune=1000, chains=4, cores=4,    # 4 chains, one per core
                  target_accept=0.9, random_seed=RANDOM_SEED)
```

This is free and you should always use it — it's how you get $\hat R$ in the first place (you *need* multiple chains to diagnose convergence). But it has a hard ceiling: it parallelizes the **number of chains**, and you rarely want more than 4–8 chains. If a *single* chain takes 9 hours, running 4 of them on 4 cores still takes 9 hours. Cross-chain parallelism cannot speed up an individual chain.

**Within-chain parallelism (Stan's `reduce_sum`).** This is the other axis: split the work of **one gradient evaluation** across threads. If your log-likelihood is a sum of conditionally independent terms — and it almost always is, $\log p(y \mid \theta) = \sum_{i=1}^N \log p(y_i \mid \theta)$ — then you can compute partial sums over chunks of the data on different threads and add them up. That speeds up *each leapfrog step*, and therefore speeds up a single chain. This is what you need when one chain is the bottleneck and you have spare cores per chain.

Stan exposes this through `reduce_sum`. PyMC does **not** have a direct `reduce_sum` equivalent in the model API — in the PyMC world, within-gradient parallelism comes from the backend (a GPU evaluating the whole vectorized sum at once, or a multi-threaded Numba/JAX kernel), not from a user-written partial-sum function. So this section is genuinely a **Stan** technique, and it's worth seeing because it's the cleanest illustration of the within-chain idea.

### 5.1 `reduce_sum` in Stan — partial-sum threading

You write a `partial_sum` function that computes the log-likelihood for a *slice* of the data, and hand it to `reduce_sum`, which farms slices out to threads and adds the results:

```stan
functions {
  // Compute the log-likelihood for one slice y[start:end] of the data.
  real partial_sum_lpmf(array[] int y_slice, int start, int end,
                        vector mu) {
    // bernoulli_logit on just this slice, using the matching slice of mu
    return bernoulli_logit_lupmf(y_slice | mu[start:end]);
  }
}
data {
  int<lower=0> N;
  array[N] int<lower=0, upper=1> y;
  matrix[N, K] X;
}
parameters {
  vector[K] beta;
}
model {
  vector[N] mu = X * beta;
  beta ~ normal(0, 1);
  int grainsize = 1;                          // 1 = let the scheduler auto-tune chunk size
  // reduce_sum splits y into slices, runs partial_sum on each in a thread, sums them
  target += reduce_sum(partial_sum_lupmf, y, grainsize, mu);
}
```

Then compile **with threading enabled** and tell CmdStanPy how many threads each chain may use:

```python
from cmdstanpy import CmdStanModel

# STAN_THREADS must be set at COMPILE time, or reduce_sum silently runs single-threaded
model = CmdStanModel(stan_file="logreg_reduce_sum.stan",
                     cpp_options={"STAN_THREADS": True})

fit = model.sample(data=stan_data,
                   parallel_chains=4,        # 4 chains in parallel (across-chain)
                   threads_per_chain=4)      # 4 threads PER chain (within-chain)
# total threads in use = parallel_chains * threads_per_chain = 16
idata = az.from_cmdstanpy(fit)              # back into the ArviZ world we know
```

> 🧮 **The math — why a summable likelihood parallelizes.** The log-posterior gradient is
> $$\nabla_\theta \log p(\theta \mid y) = \nabla_\theta \log p(\theta) + \sum_{i=1}^N \nabla_\theta \log p(y_i \mid \theta).$$
> The sum over data is the expensive part, and it's a *sum of independent terms* — so partial sums over disjoint slices $S_1, \dots, S_T$ satisfy $\sum_{i=1}^N = \sum_{t} \sum_{i \in S_t}$, and the inner sums run on separate threads with no communication until the final add. `grainsize` controls how big each slice is; `grainsize=1` lets Stan's scheduler pick, which is the right default. This is **data parallelism within one gradient**, orthogonal to running multiple chains.

> ⚠️ **Pitfall — `STAN_THREADS` at compile time, or no speedup.** The single most common `reduce_sum` mistake: forgetting `cpp_options={"STAN_THREADS": True}`. Without it, the model compiles fine, runs fine, and `reduce_sum` *silently executes single-threaded* — you get correct answers at no speedup and conclude (wrongly) that `reduce_sum` "doesn't work." The threading must be baked into the compiled binary. Total threads is `parallel_chains * threads_per_chain`, so size it to your machine: 4 chains × 4 threads = 16 threads wants 16 logical cores.

> 💡 **Intuition — the two axes are multiplicative, and you choose the split.** You have a fixed budget of CPU cores. Spend them across chains (more chains, better $\hat R$) **or** within chains (faster single chains) — and `reduce_sum` lets you do both: `parallel_chains × threads_per_chain`. If a single chain is your bottleneck and you've maxed sensible chains at 4, put the rest of your cores into `threads_per_chain`. If chains are cheap and you just want diagnostics, spend cores on `parallel_chains`. PyMC gives you the first axis (`cores`) cleanly; Stan gives you both.

---

## 6. Make the sampler's job easier: vectorize, marginalize, sparsify

The backends in §2–§5 make the *same* model run faster. This section is about changing the *model itself* so the sampler has less to do — and these structural wins are often **larger** than any backend swap, because they attack the gradient cost and the geometry at the root.

### 6.1 Vectorize — never loop in Python inside a model

This is the number-one performance trap in all of PyMC, and it's worth dwelling on. When you write a Python `for` loop that creates one random variable per iteration, PyTensor builds a *separate graph node for every iteration*. A loop over 500 groups builds 500 little `Normal` nodes, 500 little additions — a graph with thousands of tiny Ops. The compiler can't fuse them well, the gradient is a sprawl of scalar operations, and both compile and per-step time balloon. Here is the trap and the cure side by side:

```python
# ❌ SLOW — Python loop builds N separate scalar nodes
with pm.Model() as slow:
    mu = pm.Normal("mu", 0, 1)
    sigma = pm.HalfNormal("sigma", 1)
    obs = []
    for i in range(len(y)):                       # builds len(y) graph nodes!
        obs.append(pm.Normal(f"y_{i}", mu, sigma, observed=y[i]))

# ✅ FAST — one vectorized node over the whole array
with pm.Model(coords={"obs": np.arange(len(y))}) as fast:
    mu = pm.Normal("mu", 0, 1)
    sigma = pm.HalfNormal("sigma", 1)
    pm.Normal("y", mu, sigma, observed=y, dims="obs")   # ONE node, vectorized
```

The vectorized version expresses the *same* model — the same log-density — but as a single array operation that the backend compiles to one tight loop (and that a GPU can parallelize). For indexed / hierarchical structure, the vectorized idiom is **fancy indexing**, exactly as in the radon model: `a[county_idx]` selects the right intercept for every row in one operation, no loop:

```python
# hierarchical without a loop: index the group-level vector by the group index array
mu = a[county_idx] + b * floor          # vectorized; a has dims="county", result has dims="obs"
```

> ⚠️ **Pitfall — the loop you didn't notice.** The slow loop is rarely as blatant as `for i in range(len(y))`. It hides as "I'll just build the AR recursion step by step," or "I'll loop over time periods," or "I'll append to a list of likelihood terms." Any time you find yourself writing a Python loop whose body calls `pm.Something(...)` per iteration, stop — there is almost always a vectorized form (fancy indexing, `pm.math.dot`, a `pytensor.scan` for genuine recursions, or a built-in like `pm.AR` / `pm.GaussianRandomWalk` from **Chapter 15**). The profiler in §1.1 catches these: a model whose `prof.summary()` is hundreds of tiny Ops is a model with a hidden loop.

> 🔧 **In practice — `pytensor.scan` for true recursions.** Some computations are *genuinely* sequential (a state-space recursion where step $t$ needs step $t-1$). For those, the vectorized tool is `pytensor.scan`, a symbolic loop the compiler understands — not a Python loop. Better still, for the common cases (AR, random walk, structural time series, state space) reach for the **purpose-built distributions and the `pymc_extras.statespace` module from Chapter 15**, which implement the recursion (and the Kalman filter) efficiently for you.

### 6.2 Marginalize discretes and latents (recap of Ch 13 & 16)

The deepest speedup is often to **integrate a variable out of the model entirely** so the sampler never has to explore it. We met this twice already:

- **Chapter 13 (Mixtures):** instead of sampling a discrete cluster-assignment $z_i \in \{1, \dots, K\}$ for each point (which forces a Metropolis step, mixes terribly, and blocks the JAX backends), you **marginalize** $z$ analytically. The mixture likelihood sums over components: $p(y_i \mid \theta) = \sum_{k} \pi_k\, p(y_i \mid \theta_k)$, computed in log-space with `pm.math.logsumexp` (`# verify path in current PyMC docs — pm.logsumexp / pytensor.tensor.logsumexp also exist`; Stan: `log_sum_exp`). The resulting model is **fully continuous** — NUTS-able, and now any backend in §2 will take it.
- **Chapter 16 (Missing data):** a discrete missing value handled by automatic imputation triggers a fragile Metropolis step; marginalizing the discrete latent (summing over its support) recovers a clean continuous gradient.

Why is this a *speed* win and not just a correctness nicety? Two reasons. First, discrete parameters can't use NUTS at all — they get Metropolis/Gibbs, which mixes far worse, so you need many more draws for the same ESS. Second, a model that's fully continuous after marginalization can run on **any** backend, including the GPU. Marginalization simultaneously *improves mixing* and *unlocks the fast engines*. That's why it shows up in a scaling chapter.

```python
# pseudocode — the SHAPE of a marginalized 2-component mixture (full version in Chapter 13)
with pm.Model() as marg_mixture:
    w = pm.Dirichlet("w", a=np.ones(2))            # mixing weights
    mu = pm.Normal("mu", 0, 5, shape=2)
    sigma = pm.HalfNormal("sigma", 5, shape=2)
    # marginalize z: log p(y) = logsumexp_k [ log w_k + Normal_logp(y | mu_k, sigma_k) ]
    pm.Mixture("y", w=w, comp_dists=[pm.Normal.dist(mu[0], sigma[0]),
                                     pm.Normal.dist(mu[1], sigma[1])],
               observed=y)                          # pm.Mixture marginalizes z for you
    idata = pm.sample(nuts_sampler="nutpie", random_seed=RANDOM_SEED)  # continuous -> any backend
```

> 🔧 **In practice — `pm.Mixture` already marginalizes.** PyMC's `pm.Mixture` does the `logsumexp` marginalization internally, which is exactly why mixtures sample so much better in PyMC than the naive "sample the labels" approach. For more general discrete-latent marginalization, the `pymc_extras` library provides a marginalization helper (`MarginalModel` / `marginalize` — `# verify current name/location in pymc_extras docs`, it has moved between `pymc_experimental` and `pymc_extras`). In raw Stan you write the `log_sum_exp` by hand, as in **Chapter 13**.

### 6.3 HSGP and sparse approximations for GPs (recap of Ch 12)

Gaussian processes are the other place where the model structure, not the sampler, is the bottleneck. An exact GP requires factorizing an $N \times N$ covariance matrix — that's $O(N^3)$ time and $O(N^2)$ memory. At $N = 1000$ it's a few seconds; at $N = 10{,}000$ it's minutes-to-intractable. No backend swap saves you from a cubic; you must change the **approximation**.

From **Chapter 12** you have the tools:

- **HSGP** (`pm.gp.HSGP`) — the **Hilbert-Space Gaussian Process** approximation expresses the GP as a finite sum of basis functions (eigenfunctions of the Laplacian), turning the $O(N^3)$ GP into an $O(N m + m^3)$ linear model with $m$ basis functions ($m \ll N$). For a 1-D smooth GP this is dramatic — a near-exact GP at linear cost — and it's the default recommendation for scaling GPs in PyMC.
- **Sparse / inducing-point approximations** (`pm.gp.MarginalApprox`) — summarize the GP with $m$ inducing points $X_u$ (FITC/VFE/DTC), again $O(N m^2)$ instead of $O(N^3)$. (This class was `pm.gp.MarginalSparse` in PyMC v3; it was renamed to `pm.gp.MarginalApprox` in v5, so the old name will `AttributeError` — use `MarginalApprox`.)

```python
# HSGP: a scalable drop-in for a smooth 1-D GP (full treatment in Chapter 12)
with pm.Model() as hsgp_model:
    eta = pm.HalfNormal("eta", 1.0)               # amplitude
    ell = pm.InverseGamma("ell", alpha=2, beta=1) # lengthscale
    cov = eta**2 * pm.gp.cov.ExpQuad(1, ls=ell)
    gp = pm.gp.HSGP(m=[50], c=2.0, cov_func=cov)  # 50 basis functions instead of N x N
    f = gp.prior("f", X=X_gp)                      # linear-cost GP prior
    sigma = pm.HalfNormal("sigma", 1.0)
    pm.Normal("y", mu=f, sigma=sigma, observed=y_gp)
    idata = pm.sample(nuts_sampler="numpyro", random_seed=RANDOM_SEED)
```

> 💡 **Intuition — match the lever to the cost.** The cost of a model has a *dominant term*: $O(N)$ for a vanilla GLM (vectorize, minibatch, GPU), $O(N^3)$ for an exact GP (approximate it — HSGP/sparse), Metropolis-bound for a discrete latent (marginalize). Throwing a faster *sampler* at an $O(N^3)$ GP is like buying a faster car to escape a traffic jam — the constraint isn't your speed, it's the structure. Identify the dominant cost, then pick the lever that attacks *it*.

---

## 7. When the data won't fit in a gradient: minibatch ADVI

Everything so far still computes the *full-data* gradient on every step — every leapfrog touches all $N$ rows. When $N$ is in the millions, even one full gradient is too expensive to do thousands of times, and MCMC of any flavor becomes infeasible. This is where we trade exactness for scale and switch from sampling to **variational inference** (from **Chapter 06**) with a twist: estimate the gradient from a small **minibatch** of the data instead of all of it.

The idea comes straight from stochastic gradient descent. The log-likelihood is a sum over data, so a random subset of size $B$ gives an **unbiased estimate** of the full sum *if you rescale*: a minibatch of $B$ rows, multiplied by $N/B$, has the same expected value as the full sum of $N$ rows. ADVI optimizes the ELBO by stochastic gradient ascent, and a rescaled minibatch gradient is exactly the unbiased stochastic gradient it needs. So you can run ADVI seeing only a few hundred rows per step, never loading the full dataset into a single gradient.

PyMC implements this with `pm.Minibatch` (which produces index-synchronized minibatched tensors) and the `total_size` argument on the observed node (which supplies the $N/B$ rescaling):

```python
# Large-N regression we can't full-batch; fit by minibatch ADVI.
N = 1_000_000
X_big = rng.normal(size=(N, 3))
y_big = X_big @ np.array([1.5, -0.5, 2.0]) + rng.normal(size=N)

batch_size = 500
# pm.Minibatch returns minibatched, ROW-SYNCHRONIZED views of the inputs
X_mb, y_mb = pm.Minibatch(X_big, y_big, batch_size=batch_size)

with pm.Model() as big_model:
    beta = pm.Normal("beta", 0, 2, shape=3)
    sigma = pm.HalfNormal("sigma", 1)
    mu = pm.math.dot(X_mb, beta)
    # total_size = N rescales the minibatch log-likelihood by N/batch_size -> unbiased ELBO
    pm.Normal("y", mu=mu, sigma=sigma, observed=y_mb, total_size=N)

    approx = pm.fit(20000, method="advi", random_seed=RANDOM_SEED)   # returns an Approximation
    idata = approx.sample(1000)                                       # draw from the fitted q
```

Three pieces make this work, and each is a place people slip:

1. **`pm.Minibatch(*tensors, batch_size=B)`** returns minibatched versions of *all* the tensors you pass, drawing the **same** random rows from each so $X$ and $y$ stay aligned. Pass them together in one call — never minibatch $X$ and $y$ in separate calls or their rows desynchronize.
2. **`total_size=N`** on the observed node is the rescaling. Without it the ELBO is computed as if your dataset were only `batch_size` rows — the likelihood is under-weighted by a factor of $N/B$, the prior dominates, and your posterior is absurdly wide. This is the classic minibatch bug.
3. **`pm.fit(n, method="advi")`** runs the stochastic optimization and returns an `Approximation`; `approx.sample(draws)` turns the fitted variational distribution $q$ into the `InferenceData` you know. Use `method="fullrank_advi"` if the posterior has strong correlations the mean-field (diagonal) approximation would miss.

> 🧮 **The math — why $N/B$ makes the gradient unbiased.** The full-data ELBO contains $\sum_{i=1}^N \mathbb{E}_q[\log p(y_i \mid \theta)]$. Take a uniform random minibatch $S$ of size $B$. Then $\frac{N}{B}\sum_{i \in S}\log p(y_i\mid\theta)$ has expectation $\frac{N}{B}\cdot B \cdot \frac1N\sum_{i=1}^N\log p(y_i\mid\theta) = \sum_{i=1}^N\log p(y_i\mid\theta)$ — exactly the full sum. So the rescaled minibatch term is an **unbiased estimator** of the full log-likelihood term, and its gradient is an unbiased stochastic gradient of the ELBO. The $N/B$ factor is not a fudge; it's what makes minibatching correct. That factor is precisely what `total_size=N` supplies.

> ⚠️ **Pitfall — ADVI underestimates variance; don't ship it blind.** This is the **Chapter 06** warning, and it is sharper at scale because the people reaching for minibatch ADVI are exactly the people with too much data to check against MCMC. Mean-field ADVI assumes a *diagonal* Gaussian posterior — it ignores correlations and **systematically underestimates uncertainty**, sometimes badly. Your intervals will be too narrow. Before trusting a minibatch-ADVI posterior: (1) try `fullrank_advi` and see if the intervals widen (if they do, mean-field was lying); (2) if you can afford it, fit a *subsample* of the data with full NUTS and check the ADVI posterior isn't wildly off; (3) watch the ELBO trace (`approx.hist`) for convergence — a still-decreasing ELBO means you stopped too early. Minibatch ADVI is a scale tool of last resort, not a default. When MCMC is feasible, use it.

> 🩺 **Diagnostic — read the ELBO history.** `pm.fit` returns an object whose `.hist` is the ELBO (technically negative ELBO / loss) at each iteration. Plot it: `plt.plot(approx.hist)`. It should descend and **flatten** into a noisy plateau. If it's still trending down at the last iteration, you stopped too soon — raise the iteration count. If it's wildly oscillating, lower the learning rate (`obj_optimizer=pm.adam(learning_rate=...)`) or raise the batch size. A flat, noisy tail is what "converged" looks like for stochastic VI.

---

## 8. `pm.Data`: re-predict without recompiling

Here is a small habit with a large payoff, and it ties directly to the "compile vs sample" split from §1. Recall that PyMC compiles your model graph **once**. If you bake your input data in as a raw numpy array, the data is part of the graph — so to predict on *new* inputs you'd rebuild and recompile the whole model. For a big model that's a wasteful recompile every time you score new data.

The fix is `pm.Data`: a mutable container that holds your inputs *as a node you can swap*. Build the model with `pm.Data`, fit once, then `pm.set_data` to point those nodes at new inputs and predict — **no recompile**.

```python
with pm.Model(coords={"obs": np.arange(len(y_train))}) as model:
    X_shared = pm.Data("X", X_train, dims=("obs", "pred"))   # swappable input node
    beta = pm.Normal("beta", 0, 1, dims="pred")
    sigma = pm.HalfNormal("sigma", 1)
    mu = pm.math.dot(X_shared, beta)
    pm.Normal("y", mu=mu, sigma=sigma, observed=y_train, dims="obs")
    idata = pm.sample(1000, tune=1000, target_accept=0.9, random_seed=RANDOM_SEED)

# later — predict on NEW data, no recompile, just swap the container:
with model:
    pm.set_data({"X": X_new}, coords={"obs": np.arange(len(X_new))})   # resize the coord!
    preds = pm.sample_posterior_predictive(idata, predictions=True,
                                           extend_inferencedata=False, random_seed=RANDOM_SEED)
# results live in preds.predictions (kept separate from in-sample posterior_predictive)
```

> 🔧 **In practice — `pm.Data` is mutable by default in v5.** The old `pm.MutableData` / `pm.ConstantData` split is gone; plain `pm.Data` is mutable and is what you want for swap-in prediction. When the new data has a different number of rows than training, you must **resize the coordinate** in `set_data(..., coords={"obs": new_index})`, or the shapes won't line up (a known footgun — see the Chapter 15 forecasting workflow, which is the same mechanism applied to future time indices). Use `predictions=True` so out-of-sample draws land in `preds.predictions` rather than overwriting your in-sample `posterior_predictive`.

> 💡 **Intuition — separate "what's fixed" from "what's swappable."** Think of `pm.Data` as declaring *this input might change; compile around it as a slot, not a constant*. It costs nothing at fit time and saves the entire compile on every subsequent prediction. In a production setting where you fit nightly and score all day, this is the difference between a recompile per request and a free `set_data`.

---

## 9. The performance traps (the ones that quietly cost you hours)

You will hit these. Every one of them looks like "my model is just slow," and every one has a specific, unglamorous fix. Learn to recognize them by feel.

**Trap 1 — the unvectorized model (the big one).** A Python loop building one RV per iteration. Symptom: slow compile, slow sampling, `prof.summary()` is hundreds of tiny Ops. Fix: vectorize with fancy indexing / `pm.math.dot` / built-in distributions (§6.1). This is the most common and most costly trap, and the easiest to fix once you see it.

**Trap 2 — a dense mass matrix where a diagonal suffices (and the reverse).** `init="adapt_full"` learns a *full* $d \times d$ mass matrix — $O(d^2)$ memory and per-step cost. For a model with hundreds of weakly-correlated parameters, that's a huge, needless expense; the default `adapt_diag` (a diagonal metric) is far cheaper and usually fine. Conversely, if your parameters are *strongly* correlated, a diagonal metric forces long, inefficient trajectories (high `n_steps`, deep trees) — *there* a dense metric or a reparameterization to decorrelate pays off. Symptom of the wrong choice: high `n_steps` with no divergences (diagonal can't capture correlation) or surprisingly slow per-step on a big model (dense is too expensive). Match the metric to the correlation structure.

**Trap 3 — `target_accept` cranked needlessly high.** Raising `target_accept` shrinks the step size, which lengthens trajectories — `0.99` can take *several times* as many leapfrog steps per draw as `0.9`. It is the correct medicine for divergences (smaller steps navigate high-curvature regions), but if you have **zero divergences** at `0.9`, pushing to `0.99` just burns time for no benefit. Symptom: slow sampling, deep trees, clean diagnostics. Fix: drop `target_accept` back to `0.9` (the course default) — and only raise it when divergences actually appear. Don't pay the divergence-insurance premium when you have no divergences.

**Trap 4 — silent recompilation.** Re-running the model-definition cell rebuilds and recompiles the graph; you pay compile time again. In a tight edit-fit loop this dominates. Symptom: every fit has a fixed multi-second overhead even for a tiny model. Fix: build the model once, reuse it; cache compiled functions; and prefer a backend with fast compile (nutpie) while iterating.

**Trap 5 — judging a JAX backend by a short run.** XLA compile is a fixed tax that's invisible on a long run and crushing on a short one. Symptom: NumPyro "loses" on a 200-draw test. Fix: benchmark at production scale; amortize the compile over a real number of draws (§3).

**Trap 6 — too many draws.** If `az.ess` already reports ESS in the thousands, asking for 10,000 draws per chain is wasted time — you have far more effective samples than you need for stable estimates (recall the Chapter 07 guidance: ESS$_\text{bulk}$ and ESS$_\text{tail} \gtrsim 400$ is plenty for stable summaries). Symptom: long runs, enormous ESS. Fix: sample fewer draws; spend the saved time on more *models*, not more *draws of one model*.

**Trap 7 — discrete RVs blocking the fast backends.** A discrete latent forces `"pymc"` + CompoundStep (Metropolis on the discrete, which mixes poorly) and locks you out of nutpie/numpyro/blackjax. Symptom: can only use the default backend, poor ESS on the discrete part. Fix: marginalize the discrete out (§6.2) — better mixing *and* unlocks every backend.

> 🩺 **Diagnostic — the 60-second triage.** When a fit is slow, run this checklist in order: (1) `prof.summary()` — hundreds of tiny Ops? → vectorize (Trap 1). (2) `idata.sample_stats["n_steps"].mean()` high with `tree_depth` pinned at 10? → geometry: reparameterize or fix the metric (Traps 2). (3) `target_accept > 0.9` with zero divergences? → lower it (Trap 3). (4) Any discrete RVs? → marginalize (Trap 7). (5) Big $N$ and a big `Dot` dominating the profile? → GPU / minibatch (§3, §7). (6) Exact GP with $N \gtrsim$ a few thousand? → HSGP (§6.3). Each branch points at a *different* section of this chapter — that's the whole structure of scaling work.

---

## 10. The decision guide: which sampler/backend to reach for

Pin this above your desk. The choice is a function of four inputs — **does the model have discrete RVs**, **how big is the per-gradient arithmetic / data**, **what hardware do you have**, and **is exact inference (MCMC) feasible at all**.

```
START
 │
 ├─ Does the model have DISCRETE random variables you can't/won't marginalize?
 │     YES ─────────────────────────────► nuts_sampler="pymc"  (CompoundStep handles them;
 │                                          JAX/Rust backends CANNOT). Consider marginalizing
 │                                          (§6.2) to escape this branch.
 │     NO  ▼
 │
 ├─ Is full-data MCMC INFEASIBLE (N in the millions, can't full-batch a gradient)?
 │     YES ─────────────────────────────► Minibatch ADVI (pm.Minibatch + total_size, §7).
 │                                          Validate against a subsample fit. Last resort.
 │     NO  ▼
 │
 ├─ Is the dominant cost an EXACT GP (O(N^3))?
 │     YES ─────────────────────────────► HSGP / sparse approximation (§6.3), THEN pick a
 │                                          backend below for the resulting continuous model.
 │     NO  ▼
 │
 ├─ Do you have a GPU AND large per-gradient arithmetic (big N, big matrices, many chains)?
 │     YES ─────────────────────────────► nuts_sampler="numpyro",
 │                                          nuts_sampler_kwargs={"chain_method":"vectorized"}  (§3)
 │     NO  ▼
 │
 ├─ CPU, continuous model, want the fastest easy win?
 │     ───────────────────────────────► TRY nuts_sampler="nutpie" FIRST (great CPU speed,
 │                                          fast compile). Benchmark vs default "pymc" on
 │                                          ESS/second (§4). Keep the faster CONVERGED one.
 │
 └─ Need to MODIFY the algorithm (custom adaptation, SMC, tempering)?
       ───────────────────────────────► nuts_sampler="blackjax" or its low-level primitives.

ALWAYS, regardless of branch:
  • Vectorize the model first (§6.1) — no backend fixes a Python loop.
  • Fix the GEOMETRY first (non-centered, standardized, sane metric) — no backend fixes a funnel.
  • Gate on Chapter 07 diagnostics. Speed never overrides convergence.
  • Compare backends on ESS/SECOND, not wall-clock.
```

In prose, the heuristics behind the tree:

- **Discrete in the model?** Only `"pymc"` can sample it (CompoundStep). Everything else needs a continuous graph — so either stay on `"pymc"` or marginalize. This is the first question because it's a hard constraint, not a preference.
- **CPU default:** **nutpie** is the best first thing to try — fast Rust NUTS, good warmup, quick compile, often the fastest CPU option. The native `"pymc"` is the safe, maximally-compatible fallback and is competitive.
- **GPU / big vectorized models:** **NumPyro** (JAX) is the go-to — `chain_method="vectorized"` to run chains together on one device. Remember the GPU only helps when each gradient is large (§3); a small model is faster on CPU.
- **Research / custom algorithms:** **BlackJAX** — composable JAX primitives when you want to change the sampler, not just run it.
- **Too much data for any MCMC:** **minibatch ADVI** — accept the variational approximation's narrower intervals, and validate.
- **GP at scale:** approximate the GP (**HSGP**) *before* choosing a backend; the cubic cost is structural and no engine escapes it.

> 🔧 **In practice — the realistic workflow.** Ninety percent of the time the answer is: write a *vectorized, non-centered* model; sample with the default `"pymc"`; if it's slow, add `nuts_sampler="nutpie"` and re-run; if *that's* slow and the model is big with a GPU available, switch to `nuts_sampler="numpyro"` with `chain_method="vectorized"`. Minibatch ADVI, `reduce_sum`, and BlackJAX are specialist tools for specific corners (truly huge data, Stan within-chain threading, algorithm research). Reach for the simple lever first; escalate only when the profile (§1) tells you to.

---

## 11. Bringing it together: a scaling worked example end-to-end

Let me close the practical arc by walking the *whole decision* on one model, so you see the chapter as a single workflow rather than a menu. We'll scale up the radon model from §4 — imagine the dataset grew to many thousands of houses across more counties, and the default fit now takes uncomfortably long.

**Step 0 — profile.** Time the build vs the sample (§1.1); read `sample_stats`. Suppose we find: compile is a few seconds (fine), but `n_steps` averages ~200 and `tree_depth` sometimes hits 10 — a *geometry* smell. Divergences: a handful.

**Step 1 — fix the model, not the sampler.** The handful of divergences and deep trees say funnel. We're already non-centered (§4.2), good — but we double-check standardization (predictors on unit scale, the **Chapter 03** default) so the mass matrix has an easy job, and confirm there's no hidden Python loop (`prof.summary()` is a few big Ops, not hundreds of tiny ones — clean). Geometry handled.

```python
# confirm geometry is healthy before optimizing speed
print("n_steps mean:", float(idata.sample_stats["n_steps"].mean()))
print("max tree_depth:", int(idata.sample_stats["tree_depth"].max()))
print("divergences:", int(idata.sample_stats["diverging"].sum()))
# if divergences remain after non-centering: bump target_accept to 0.95, re-profile
```

**Step 2 — try the cheap backend swap.** No discrete RVs, CPU machine → try nutpie first:

```python
with radon_model:
    idata_nutpie = pm.sample(1000, tune=1000, chains=4, target_accept=0.9,
                             nuts_sampler="nutpie", random_seed=RANDOM_SEED)
# compare ESS/second vs the default fit (§4.3). Keep the faster converged one.
```

**Step 3 — escalate only if needed.** If the dataset were genuinely large (hundreds of thousands of rows) and a GPU were available, switch to NumPyro with vectorized chains:

```python
with radon_model:
    idata_gpu = pm.sample(1000, tune=1000, chains=4, target_accept=0.9,
                          nuts_sampler="numpyro",
                          nuts_sampler_kwargs={"chain_method": "vectorized"},
                          random_seed=RANDOM_SEED)
```

**Step 4 — gate and ship.** Whatever backend won, confirm the **Chapter 07** diagnostics ($\hat R \le 1.01$, ESS$_\text{bulk}$/ESS$_\text{tail} \gtrsim 400$, zero divergences, BFMI $> 0.3$) and that the posterior matches the default's to Monte Carlo error. Then close the workflow loop exactly as in §4: a **posterior predictive check** (`pm.sample_posterior_predictive(idata, extend_inferencedata=True)` + `az.plot_ppc`) to confirm the *model* still fits the data — speed never excuses skipping the predictive check. Finally use `pm.Data` + `set_data` (§8) to score new counties without recompiling. Done — same statistics, a fraction of the wall-clock, and you can *explain every choice you made*.

That sequence — **profile → fix geometry → cheap backend swap → escalate by the profile → gate** — is the transferable skill. It generalizes to every slow model you will ever meet.

---

## ⚠️ Common errors & how to fix them

| Symptom / message | Cause | Fix |
|---|---|---|
| `numpyro` / `blackjax` / `nutpie` **import error** | JAX/Rust backend not installed | `pip install numpyro` (or `blackjax`, `nutpie`); for GPU install `jax[cuda12]` per JAX docs |
| Backend chosen but model has a `pm.DiscreteUniform` / `pm.Categorical` → **"cannot sample"** | JAX/Rust backends are pure-NUTS; they can't do discrete RVs | Use `nuts_sampler="pymc"` (CompoundStep), **or marginalize** the discrete out (§6.2) so the graph is continuous |
| nutpie **JAX mode fails** / falls back to slow | JAX deps missing or backend not set | `nuts_sampler_kwargs={"backend":"jax","gradient_backend":"jax"}` and ensure `jax` is installed |
| GPU fit is **slower** than CPU | Small per-gradient arithmetic; sequential MCMC can't use the cores; transfer/kernel-launch overhead dominates | GPU only helps big vectorized models / many chains (§3). Use CPU (nutpie) for small models |
| JAX backend "loses" on a **short** benchmark | XLA **compile time** dominates a short run | Benchmark at production scale; compile is amortized over a long run (§3) |
| Minibatch ADVI posterior **absurdly wide / wrong scale** | Forgot `total_size=N` → likelihood under-weighted by $N/B$ | Add `total_size=N` to the observed node (§7) |
| `pm.Minibatch` gives **misaligned** X and y | Minibatched X and y in **separate** calls → desynced rows | Pass all tensors in one call: `pm.Minibatch(X, y, batch_size=B)` |
| Minibatch-ADVI intervals **too narrow** | Mean-field ADVI underestimates variance (Ch 06) | Try `method="fullrank_advi"`; validate vs a subsample NUTS fit; check `approx.hist` converged |
| Model **slow to compile and sample**, `prof.summary()` is hundreds of tiny Ops | Unvectorized: a Python loop building one RV per iteration | Vectorize: fancy indexing `a[idx]`, `pm.math.dot`, built-in dists, `pytensor.scan` for true recursions (§6.1) |
| High `n_steps` / `tree_depth` pinned at 10, **no divergences** | Poorly-scaled metric or genuine correlation the diagonal can't capture | Standardize; `init="jitter+adapt_full"` (dense metric) **or** reparameterize to decorrelate |
| Per-step **very slow on a big model** with `adapt_full` | Dense $d\times d$ mass matrix is $O(d^2)$ | Use default `adapt_diag` unless parameters are strongly correlated (Trap 2) |
| Sampling slow, deep trees, **zero divergences**, `target_accept` high | `target_accept` set needlessly high (e.g. 0.99) | Drop back to `0.9`; only raise it when divergences actually appear (Trap 3) |
| Exact GP intractable at `N` in the thousands | $O(N^3)$ covariance factorization | Use `pm.gp.HSGP` (Hilbert-space) or sparse `MarginalApprox` (§6.3) — no backend escapes the cubic |
| `reduce_sum` (Stan) gives **no speedup** | Model compiled without threading | `cpp_options={"STAN_THREADS": True}` at compile **and** `threads_per_chain>1` at sample |
| Re-predicting **recompiles** the whole model every time | Inputs baked in as raw arrays, part of the graph | Wrap inputs in `pm.Data`; swap with `pm.set_data` (§8) — no recompile |
| Forecast / prediction **shape error** after `set_data` | New data has different row count; coord not resized | `pm.set_data({...}, coords={"obs": new_index})` and use `predictions=True` |
| Every tiny fit has fixed multi-second overhead | Silent **recompilation** from re-running the model cell | Build the model once, reuse it; use a fast-compile backend (nutpie) while iterating (Trap 4) |

---

## 🧪 Exercises

**1 (conceptual) — Localize the slowness.** For each scenario, name which of the four time-sinks from §1 dominates and which section's tool you'd reach for: (a) a 5-parameter regression on 2 million rows; (b) a centered hierarchical model with a funnel throwing 300 divergences; (c) an exact GP on 8,000 points; (d) a mixture model where you're sampling the cluster labels directly; (e) a tiny model you're refitting 50 times while editing priors. *Hint: gradient cost, steps-per-draw, model structure, and compile each point at a different fix — and none of them is "buy a GPU" for most of these.*

**2 (coding) — Race the backends honestly.** Take the radon model from §4. Fit it with `nuts_sampler` set to `"pymc"`, `"nutpie"`, and (if installed) `"numpyro"`. For each, record wall-clock **and** minimum ESS across the population parameters, and compute ESS/second. Which wins? Now scale the data: tile the dataset 50× (`np.tile`) and re-race. Does the ranking change? *Hint: the JAX compile tax matters less as the run gets longer — that's the §3 lesson made numerical.*

**3 (coding) — The `total_size` bug, on purpose.** Fit the large-$N$ regression from §7 with minibatch ADVI **twice**: once correctly with `total_size=N`, once *omitting* `total_size`. Compare the posterior for `beta` between the two and against the known true coefficients. By roughly what factor are the intervals wrong without `total_size`, and why is it ~$N/B$? *Hint: without the rescaling the likelihood thinks you have only `batch_size` data points, so the prior dominates.*

**4 (coding) — Vectorize a loop.** Write the "slow" looped model from §6.1 (one `pm.Normal` per observation) and the vectorized version, on 500 data points. Profile both with `model.profile(model.logp()).summary()` and compare the number of Ops and the total time. Then time a full `pm.sample` for each. How large is the gap, and where does it come from — compile, per-step, or both? *Hint: count the Ops in each profile; the looped model's count scales with N.*

**5 (conceptual) — Two axes of parallelism.** You have a 16-core machine and a single chain that takes 2 hours. Explain why `cores=16` does **not** make that chain finish faster, and describe how Stan's `reduce_sum` *would* help. Then compute the total threads for `parallel_chains=4, threads_per_chain=4` and explain what each axis buys you statistically (hint: one axis gives you $\hat R$, the other gives you speed per chain).

**6 (coding, stretch) — `pm.Data` re-prediction.** Fit the radon model with `pm.Data` for the floor and county-index inputs. Then, *without recompiling*, use `pm.set_data` to predict log-radon for a brand-new set of houses (mix of basement and first-floor across several existing counties), and report posterior predictive intervals via `predictions=True`. Confirm (e.g. by timing) that the second prediction did **not** trigger a recompile. *Hint: remember to resize the `"obs"` coord in `set_data`.*

---

## 📚 Resources & further reading

**The backend & speed how-tos (start here):**
- **"Faster Sampling with JAX and Numba"** — PyMC example gallery — the official benchmark/how-to and the page to cite for *how to pick a backend*. https://www.pymc.io/projects/examples/en/latest/samplers/fast_sampling_with_jax_and_numba.html
- **"Other NUTS Samplers"** — pymc-marketing docs — side-by-side `numpyro` / `nutpie` / `blackjax` usage with install notes. https://www.pymc-marketing.io/en/stable/notebooks/general/other_nuts_samplers.html
- **`pymc.sample` API** — confirm `nuts_sampler`, `nuts_sampler_kwargs`, `compile_kwargs` spellings and defaults. https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.sample.html

**The backends themselves:**
- **NumPyro docs** — JAX-based PPL; NUTS, `chain_method`, GPU. https://num.pyro.ai
- **BlackJAX docs** — composable JAX inference primitives (NUTS, window adaptation, SMC, Pathfinder). https://blackjax-devs.github.io/blackjax/
- **nutpie** — fast Rust NUTS for PyMC/Stan; Numba and JAX backends. https://github.com/pymc-devs/nutpie
- **JAX install guide** — the authoritative, frequently-changing `jax[cuda12]` install instructions. https://jax.readthedocs.io/en/latest/installation.html

**Within-chain parallelism (Stan):**
- **Stan `reduce_sum` tutorial** — minimal `partial_sum` + `reduce_sum` walkthrough. https://mc-stan.org/learn-stan/case-studies/reduce_sum_tutorial.html
- **CmdStan Parallelization guide** — `STAN_THREADS`, `threads_per_chain`, total-thread math. https://mc-stan.org/docs/cmdstan-guide/parallelization.html

**Minibatch VI & GPs at scale:**
- **"GLM: Mini-batch ADVI on hierarchical regression"** — the canonical `pm.Minibatch` + `total_size` example. https://www.pymc.io/projects/examples/en/latest/variational_inference/GLM-hierarchical-advi-minibatch.html
- **PyMC GP documentation & gallery** — the Gaussian-process docs index (kernels, `pm.gp.HSGP` for the Hilbert-space approximation, `pm.gp.MarginalApprox` for inducing-point sparse GPs) for scaling GPs (recap of Ch 12). https://www.pymc.io/projects/docs/en/stable/api/gp.html  (the HSGP notebook lives in the examples gallery under `gaussian_processes/` — confirm the current slug there)

**The foundations these tools rest on:**
- **Hoffman & Gelman, "The No-U-Turn Sampler," JMLR 15 (2014)** — the NUTS algorithm every backend implements. https://jmlr.org/papers/v15/hoffman14a.html
- **Betancourt, "A Conceptual Introduction to Hamiltonian Monte Carlo," arXiv:1701.02434** — *why* gradients let HMC scale to high dimensions where random-walk MCMC dies. https://arxiv.org/abs/1701.02434
- **Hoffman et al., "Automatic Differentiation Variational Inference," JMLR (2017)** — the ADVI that minibatching builds on. https://jmlr.org/papers/v18/16-107.html
- **Martin, Kumar, Lao, *Bayesian Modeling and Computation in Python*** — free online; Ch 2 & 11 cover inference engines, VI, and practical computation in PyMC/ArviZ. https://bayesiancomputationbook.com
- **A Primer on Bayesian Methods for Multilevel Modeling (radon)** — the definitive radon load (`pm.get_data("srrs2.dat")` / `cty.dat`) and processing used in §4. https://www.pymc.io/projects/examples/en/latest/generalized_linear_models/multilevel_modeling.html
- **PyMC discourse** — the practical place to ask "which backend for my model?" with real benchmarks from the community. https://discourse.pymc.io

---

## ➡️ What's next

You now have the full computational toolkit: the geometry from **Chapters 05 & 07**, the model structure from **Chapters 09–16**, and — as of this chapter — the ability to make all of it run fast and at scale without changing a single statistical conclusion. That completes the machinery half of the course. **Chapter 18 — Capstone Case Studies** puts everything together: two or three end-to-end projects on real datasets that chain the *entire* Bayesian workflow — prior predictive checks, fitting (with the right backend chosen deliberately, using exactly the decision guide from §10), diagnostics, posterior predictive checks, model comparison, and honest communication of uncertainty. You'll watch a slow model get profiled and accelerated, a GP get HSGP-approximated, a discrete latent get marginalized so a JAX backend can take it — the scaling reflexes from this chapter woven into the workflow, where they belong. The golem is built and it's fast; now we set it loose on real problems, start to finish.
