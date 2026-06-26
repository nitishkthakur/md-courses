# Chapter 12 — Gaussian Processes

### Putting a prior on *functions* — the Bayesian way to learn an unknown curve without committing to a parametric form

> *A note from me to you. Up to now, every model we built had a fixed skeleton: a line, a logit, a Poisson rate driven by a handful of named coefficients. We chose the shape of the function in advance and let the data fill in the numbers. But what do you do when you genuinely do not know the shape? When the relationship might be smooth-and-gentle here, sharp-and-wiggly there, and you do not want to pretend a cubic polynomial captures it? The lazy answer is "add more features and pray." The Bayesian answer — and it is one of the most beautiful ideas in this whole course — is to put a prior directly on the space of functions, and let the posterior tell you which functions are plausible given the data. That object is a Gaussian process. It is the most honest nonparametric regression you can do: the model is allowed to be as flexible as the data demand and no more, and it comes with calibrated uncertainty that fans out exactly where you have no data and pinches in exactly where you do. The painful mistake this chapter prevents is twofold. First: overfitting yourself into a corner with a too-flexible model that has no principled way to say "I don't know here." Second — and this one bites every newcomer — fitting a GP whose lengthscale prior is flat, watching it sample for an hour, and getting a wall of divergences because the geometry is pathological. By the end you will know why that happens and exactly which prior fixes it. Sit with this chapter. GPs reward patience, and they will change how you think about flexible modeling forever.*

---

## What you'll be able to do after this chapter

- Explain a Gaussian process as a **distribution over functions**, and translate that into the operational "multivariate normal evaluated at the data points" view you can actually compute with.
- Read and choose **kernels** — ExpQuad/RBF, Matérn-3/2 and Matérn-5/2, Periodic, Linear — and know exactly what the **lengthscale** $\ell$, **amplitude** $\eta$, and **noise** $\sigma$ each control.
- **Compose** kernels by addition and multiplication to build structured priors (trend × periodic seasonal + short-scale noise).
- Fit **GP regression** with a Gaussian likelihood using `pm.gp.Marginal`, and produce posterior predictive curves with credible bands using `.conditional`.
- Fit **latent GPs** for non-Gaussian likelihoods (classification, Poisson counts) with `pm.gp.Latent`.
- Put a **principled prior on the lengthscale** (InverseGamma / Gamma) and explain Betancourt's argument for why a too-short lengthscale both overfits *and* wrecks your sampler.
- Beat the **$O(n^3)$ scaling wall** with the **HSGP** Hilbert-space approximation (`pm.gp.HSGP`) and sparse inducing-point methods (`pm.gp.MarginalApprox`), and size `m` and `c` sensibly.
- Decompose the **Mauna Loa CO₂** series into trend + seasonal + noise with an additive kernel and predict each component.
- Place GPs in the broader landscape: how they connect to **splines**, **BART**, and **infinite-width neural networks**.

---

## 1. The leap: from a prior on numbers to a prior on functions

Let me build this the way McElreath would — slowly, from a place you already trust.

You already know the multivariate normal. If I hand you a mean vector $\boldsymbol\mu \in \mathbb{R}^n$ and a covariance matrix $\Sigma \in \mathbb{R}^{n\times n}$, then

$$
\mathbf{f} = (f_1, f_2, \dots, f_n) \sim \mathcal{N}(\boldsymbol\mu, \Sigma)
$$

is a recipe for drawing $n$ correlated numbers. The diagonal of $\Sigma$ says how much each $f_i$ varies on its own; the off-diagonals say how much pairs move together. Nothing new there.

Now make one conceptual move, and everything opens up. Attach an **input location** $x_i$ to each component $f_i$. Think of $f_i$ as "the value of some unknown function $f$ evaluated at input $x_i$": $f_i = f(x_i)$. Suddenly the vector $\mathbf{f}$ is not just a bag of numbers — it is the function $f$ *sampled at the points* $x_1, \dots, x_n$. If I connect the dots, I get a curve.

> 💡 **Intuition:** A multivariate normal is a distribution over **vectors**. A Gaussian process is what you get when you let the index of the vector be a **continuous input** $x$ instead of an integer $i$. It is a multivariate normal with infinitely many components — one for every possible input value — and any finite slice of it (the values at any finite set of inputs) is an ordinary multivariate normal. That "any finite slice is Gaussian" property is the entire definition.

Here is the formal statement, and it is shorter than you'd fear:

> 🧮 **The math — GP definition.** A function $f$ is distributed as a **Gaussian process** with mean function $m(x)$ and covariance (kernel) function $k(x, x')$, written
> $$ f \sim \mathcal{GP}\big(m(x),\, k(x, x')\big), $$
> if and only if, for **any** finite collection of inputs $X = \{x_1, \dots, x_n\}$, the vector of function values is multivariate normal:
> $$ \mathbf{f} = \big(f(x_1), \dots, f(x_n)\big) \sim \mathcal{N}\big(\mathbf{m}_X,\, K_{XX}\big), $$
> where $\mathbf{m}_X = \big(m(x_1), \dots, m(x_n)\big)$ and $(K_{XX})_{ij} = k(x_i, x_j)$.

That's it. The GP is **fully specified** by two functions:

- the **mean function** $m(x)$ — your prior guess for the average value of $f$ at each input, before seeing data;
- the **covariance / kernel function** $k(x, x')$ — which encodes how strongly $f(x)$ and $f(x')$ co-vary, i.e. how *similar* you expect the function to be at two nearby (or far-apart) inputs.

The kernel is where all the modeling happens. It is, quite literally, your prior belief about smoothness, wiggliness, periodicity, and scale — written as a function of two inputs.

### The operational view (the one you compute with)

The infinite-dimensional story is gorgeous, but here is the part that makes GPs *practical*: **you never need the infinite object.** You only ever evaluate $f$ at finitely many places — the $n$ training inputs and the $n_*$ prediction inputs. So the only thing you ever instantiate is a finite multivariate normal with a covariance matrix built by evaluating $k$ on those points.

In ASCII, building the training covariance matrix from the kernel:

```
        x_1   x_2   x_3  ...  x_n
x_1 [ k(1,1) k(1,2) k(1,3) ... k(1,n) ]
x_2 [ k(2,1) k(2,2) k(2,3) ... k(2,n) ]
x_3 [ k(3,1) k(3,2) k(3,3) ... k(3,n) ]   = K_XX   (n x n, symmetric, PSD)
 .  [   .      .      .          .    ]
x_n [ k(n,1) k(n,2) k(n,3) ... k(n,n) ]
```

That matrix $K_{XX}$ **is** your prior over the function values at the data. Everything else — fitting, predicting, uncertainty — is multivariate-normal algebra on this matrix. When people say "a GP is just a multivariate normal with a structured covariance evaluated at the data points," *that is the entire operational content of the method*. Internalize it and the rest is bookkeeping.

> 🔧 **In practice:** This also tells you immediately where the cost lives. The matrix is $n \times n$. To fit and predict you must factorize it (a Cholesky decomposition) and solve linear systems — both $O(n^3)$. That cubic cost is the GP's original sin, and §10 is entirely about defeating it.

### What does a prior draw *look like*?

Before any data, a GP is a cloud of functions. To draw one sample function, you pick a grid of inputs $X$, build $K_{XX}$ from your kernel, build the mean vector $\mathbf{m}_X$ (often just zeros), and draw one vector from $\mathcal{N}(\mathbf{m}_X, K_{XX})$. Connect the dots — that's one plausible function from your prior. Draw again — another. The *spread* of these curves at each $x$ is your prior uncertainty; the *texture* (smooth vs jagged, flat vs wavy) is dictated entirely by the kernel.

Here is code that draws prior functions from a GP so you can *see* the prior before we ever fit anything:

```python
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm

RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)

# A grid of inputs. NOTE: GP inputs in PyMC must be 2-D, shape (n, 1) for 1-D input.
X = np.linspace(0, 10, 200)[:, None]

# Build a kernel: amplitude eta=1, lengthscale ell=1, exponentiated-quadratic.
eta, ell = 1.0, 1.0
cov_func = eta**2 * pm.gp.cov.ExpQuad(input_dim=1, ls=ell)

# Evaluate the covariance matrix on the grid and add tiny jitter for numerical PSD-ness.
K = cov_func(X).eval()            # .eval() turns the PyTensor expression into a numpy array
K += 1e-8 * np.eye(X.shape[0])

# Draw 6 prior functions (mean = 0).
prior_draws = rng.multivariate_normal(mean=np.zeros(X.shape[0]), cov=K, size=6)

plt.plot(X[:, 0], prior_draws.T)
plt.title("Six functions drawn from a GP prior (ExpQuad, eta=1, ell=1)")
plt.xlabel("x"); plt.ylabel("f(x)")
```

When you run this, you will see six smooth, gently undulating curves, each wandering within roughly $\pm 2$ of zero (because $\eta = 1$, so the marginal standard deviation of $f$ at any point is $1$, and $\pm 2$ is the 95% band). They wiggle on a horizontal scale of about $\ell = 1$ input unit. **Change `ell` to 0.2** and re-draw: the curves become frantic and jagged, decorrelating over a fifth of the distance. **Change `ell` to 4** and they become long lazy swells. **Change `eta` to 3** and the vertical spread triples while the wiggle rate stays the same. Spend five minutes turning these two knobs and watching the prior; it builds more intuition than any equation.

> ⚠️ **Pitfall:** PyMC's GP machinery requires the input array `X` to be **2-D**, shape `(n, 1)` even for a single input dimension. Passing a flat `(n,)` array is the single most common GP error and produces a confusing shape error from inside the covariance function. Always `X = x[:, None]`.

---

## 2. Kernels: the grammar of structure

The kernel $k(x, x')$ is the soul of a GP. It must be a valid covariance function, which means: for **any** finite set of inputs, the resulting matrix $K_{XX}$ must be symmetric and positive semi-definite (PSD) — otherwise it is not a legal covariance and the multivariate normal is undefined. The wonderful news is that you rarely have to check this by hand: there is a small library of building-block kernels that are PSD by construction, and they stay PSD under addition and multiplication. So you compose.

Let me walk through the kernels you will actually use, what each one *means*, and what its draws look like.

### 2.1 ExpQuad / RBF / squared-exponential — the smooth default

The exponentiated-quadratic kernel (PyMC calls it `ExpQuad`; the ML literature calls it RBF or the squared-exponential or Gaussian kernel — all the same thing) is

$$
k(x, x') = \eta^2 \exp\!\left(-\frac{\lVert x - x' \rVert^2}{2\ell^2}\right).
$$

> 💡 **Intuition:** Similarity decays with the **square** of the distance between inputs. Two inputs closer than one lengthscale are strongly correlated; beyond two or three lengthscales they are essentially independent. The $\eta^2$ out front is the variance scale — it sets how big the function's deviations from the mean can be.

ExpQuad draws are **infinitely differentiable** — buttery smooth, no kinks anywhere. That sounds great, and it is the reason ExpQuad is everyone's first kernel. But it is also a known criticism: real-world functions are rarely *that* smooth. ExpQuad assumes the function and *all* its derivatives are continuous, which is a strong, often unrealistic, prior. It tends to produce overconfident extrapolations and can fight the data when the truth has sharper features.

### 2.2 Matérn — realistic roughness

The Matérn family relaxes the smoothness. It has a parameter $\nu$ controlling differentiability: the function is $\lceil \nu \rceil - 1$ times differentiable. The two you will use constantly are $\nu = 3/2$ and $\nu = 5/2$. With $r = \lVert x - x' \rVert$:

$$
k_{3/2}(r) = \eta^2\left(1 + \frac{\sqrt{3}\,r}{\ell}\right)\exp\!\left(-\frac{\sqrt{3}\,r}{\ell}\right)
\qquad\text{(once differentiable)}
$$

$$
k_{5/2}(r) = \eta^2\left(1 + \frac{\sqrt{5}\,r}{\ell} + \frac{5r^2}{3\ell^2}\right)\exp\!\left(-\frac{\sqrt{5}\,r}{\ell}\right)
\qquad\text{(twice differentiable)}
$$

> 💡 **Intuition:** Matérn-3/2 draws are continuous but visibly "rougher" — they have kinks, like a sample path of a physical process buffeted by noise. Matérn-5/2 sits in between: smoother than 3/2, less unrealistically perfect than ExpQuad. **Matérn-5/2 is the pragmatic default for most real regression problems.** As $\nu \to \infty$, Matérn converges to ExpQuad — so think of ExpQuad as the infinitely-smooth limit of the Matérn family.

> 📜 **Citation/Origin:** This "ExpQuad is too smooth, prefer Matérn for realistic functions" point is made forcefully in Rasmussen & Williams (*Gaussian Processes for Machine Learning*, Ch. 4) and is folklore among practitioners. Stein's *Interpolation of Spatial Data* (1999) argues the same from a spatial-statistics angle: assuming infinite differentiability is usually unwarranted.

### 2.3 Periodic — for seasonality and cycles

When you know the function repeats — daily, weekly, yearly — you want a kernel that makes $f(x)$ and $f(x + p)$ correlated for a period $p$. The periodic (a.k.a. exp-sine-squared) kernel is

$$
k(x, x') = \eta^2 \exp\!\left(-\frac{2\sin^2\!\big(\pi |x - x'| / p\big)}{\ell^2}\right).
$$

The `period` $p$ sets the cycle length; the lengthscale $\ell$ now controls **the wiggliness *within* one period** — small $\ell$ means a complicated, many-harmonic shape repeats; large $\ell$ means a smooth near-sinusoid repeats.

### 2.4 Linear — Bayesian linear regression in disguise

$$
k(x, x') = \eta^2 \, (x - c)(x' - c)
$$

A GP with a linear kernel produces **straight-line** draws — it *is* Bayesian linear regression, viewed through the GP lens. On its own it is boring, but it shines in compositions: multiply a linear kernel by another kernel to get a function whose amplitude grows with $x$, or add a linear kernel to capture a global trend on top of local wiggles.

### 2.5 Composing kernels — the real power

Here is the move that makes GPs a *modeling language* rather than a single trick. Valid kernels are closed under addition and multiplication:

- **Sum** $k_1 + k_2$: the function is a **sum of independent components**, one drawn from each kernel. Use this for "trend PLUS seasonality PLUS noise." Additive structure.
- **Product** $k_1 \times k_2$: the components **interact / modulate** each other. The classic use is `Periodic × ExpQuad` to get a *decaying* periodic signal — a seasonal pattern that is allowed to slowly change its shape over years rather than repeating identically forever.

> 🧮 **The math — why sums and products stay valid.** If $K_1$ and $K_2$ are PSD matrices, so are $K_1 + K_2$ (sum of PSD is PSD) and $K_1 \odot K_2$ (the elementwise/Hadamard product — Schur product theorem). So any sum or product of valid kernels is a valid kernel. That is the entire algebraic foundation of "kernel composition." You can build remarkably expressive priors with nothing but `+` and `*`.

In PyMC the composition is just Python operators on covariance objects:

```python
# A trend + a decaying-seasonal + short-scale noise, in one expression:
cov_trend    = eta_t**2 * pm.gp.cov.Matern52(input_dim=1, ls=ell_t)
cov_season   = eta_s**2 * pm.gp.cov.Periodic(input_dim=1, period=1.0, ls=ell_s) \
                        * pm.gp.cov.ExpQuad(input_dim=1, ls=ell_decay)   # decay envelope
cov_noise    = eta_n**2 * pm.gp.cov.Matern32(input_dim=1, ls=ell_n)

cov_func = cov_trend + cov_season + cov_noise
```

Each operator returns a new `Covariance` object, so the whole thing composes cleanly. Multiplying an amplitude in as `eta**2 * cov` is itself a composition (scalar × kernel).

> 🔧 **In practice — the PyMC kernel catalog.** The covariance functions live in `pm.gp.cov.*`: `ExpQuad`, `Matern12`, `Matern32`, `Matern52`, `Exponential`, `RatQuad` (rational quadratic, a scale-mixture of ExpQuads with extra parameter `alpha`), `Periodic`, `Cosine`, `Linear`, `Polynomial`, `WhiteNoise(sigma)`, `Constant`, `Gibbs` (nonstationary), `Coregion`, `Kron`. Mean functions live in `pm.gp.mean.*`: `Zero`, `Constant`, `Linear`. Constructors are `pm.gp.cov.ExpQuad(input_dim, ls=...)` and `pm.gp.cov.Periodic(input_dim, period=..., ls=...)`. Use `period=` and `ls=` as keywords to be safe across versions.

Here is a compact table to keep on your desk:

| Kernel | Formula (core) | Smoothness of draws | Reach for it when… |
|---|---|---|---|
| `ExpQuad` (RBF) | $\exp(-r^2/2\ell^2)$ | $C^\infty$ (infinitely smooth) | function is genuinely very smooth; default starting point |
| `Matern52` | poly·$\exp(-\sqrt5 r/\ell)$ | twice differentiable | **realistic default** for most regression |
| `Matern32` | poly·$\exp(-\sqrt3 r/\ell)$ | once differentiable | rougher, physical-process-like signals, noise components |
| `Periodic` | $\exp(-2\sin^2(\pi r/p)/\ell^2)$ | depends on $\ell$ | known cycle/season of period $p$ |
| `Linear` | $(x-c)(x'-c)$ | straight lines | global trend; multiply to modulate |
| `RatQuad` | $(1 + r^2/2\alpha\ell^2)^{-\alpha}$ | smooth, multi-scale | mixture of lengthscales in one component |
| `WhiteNoise` | $\sigma^2\,\delta_{x,x'}$ | independent jitter | explicit i.i.d. noise on the diagonal |

---

## 3. What the knobs actually control (drill this until it's reflex)

Three numbers govern almost every GP you will fit. Misunderstanding them is the root of most GP frustration, so let me be very explicit.

**Lengthscale $\ell$ — the wiggliness knob.** It is the input-distance over which the function "forgets" itself, i.e. over which $f(x)$ and $f(x')$ decorrelate.

- *Small* $\ell$ → the function decorrelates over a tiny distance → it can wiggle wildly between nearby points → **flexible, and dangerously prone to overfitting**: with a small enough $\ell$ a GP can thread a smooth curve through literally every data point, noise and all.
- *Large* $\ell$ → the function changes only over long distances → nearly flat / very smooth → **rigid, prone to underfitting**.

**Amplitude $\eta$ (sometimes written `eta`) — the vertical-scale knob.** It is the marginal standard deviation of the function: at any single input, $f(x)$ has standard deviation $\eta$ around the mean. So $\eta$ controls how far the function is allowed to swing up and down. It does **not** affect wiggliness — only vertical reach. A function with $\eta = 5$ makes the same shapes as $\eta = 1$, just five times taller.

**Noise $\sigma$ — the observation-noise knob (regression only).** This is *not* a property of $f$; it is the i.i.d. measurement noise added to the *observations*: $y_i = f(x_i) + \varepsilon_i$, $\varepsilon_i \sim \mathcal{N}(0, \sigma^2)$. In the math it sits on the diagonal of the covariance: $(K_{XX} + \sigma^2 I)$. It is what lets the GP *not* interpolate every point — it tells the model "some of the scatter you see is junk, don't chase it."

> ⚠️ **Pitfall — the lengthscale/noise tug-of-war.** A GP can explain a scattered cloud of data in two opposite ways: (a) a *smooth* $f$ (largish $\ell$) plus *large* noise $\sigma$, or (b) a *wiggly* $f$ (tiny $\ell$) plus *tiny* noise that threads every point. Both fit the training data; they generalize completely differently. The wiggly-with-no-noise explanation is overfitting, and — crucially — it lives in a flat, badly-behaved region of the posterior that destroys your sampler. The cure is a sensible prior on $\ell$, which is exactly the subject of §6. Keep this tension in mind; it is *the* central difficulty of practical GP modeling.

> 🩺 **Diagnostic — reading the knobs from posterior summaries.** After fitting, run `az.summary(idata, var_names=["ell", "eta", "sigma"])`. If the posterior for `ell` has piled up near zero (and `sigma` near zero), your GP is overfitting — it found the "wiggle through everything" solution. If `ell` is enormous relative to your input range, the GP has decided the function is basically flat. Either extreme is a signal to revisit your prior or your kernel choice.

---

## 4. GP regression with a Gaussian likelihood — `pm.gp.Marginal`

We now make the GP *do work*. The most common and most tractable case: continuous outcome $y$, Gaussian observation noise. The generative story:

$$
f \sim \mathcal{GP}(m, k), \qquad y_i \mid f \sim \mathcal{N}\big(f(x_i),\, \sigma^2\big).
$$

The magic of the Gaussian likelihood is that we can **integrate out the infinite-dimensional $f$ analytically**. A Gaussian prior on $f$ plus a Gaussian likelihood gives a Gaussian marginal for $y$, and a Gaussian conditional for predictions. No need to sample the latent function values at all during fitting — you sample only the *hyperparameters* $(\ell, \eta, \sigma)$, which is a tiny, well-behaved space. This is why `pm.gp.Marginal` exists and why it is so much faster than the latent version we'll meet in §5.

> 🧮 **The math — the two equations that define GP regression.** Marginalizing the latent function, the predictive distribution of $f$ at new inputs $X_*$ is Gaussian with
> $$ \bar{f}_* = K_{*X}\,(K_{XX} + \sigma^2 I)^{-1}\,y, $$
> $$ \mathrm{cov}(f_*) = K_{**} - K_{*X}\,(K_{XX} + \sigma^2 I)^{-1}\,K_{X*}. $$
> The first line is the posterior **mean** prediction — a smooth interpolation of the data, weighted by kernel similarity. The second is the posterior **covariance** — and notice it *shrinks* near data (the subtracted term is large there) and *returns to the prior* $K_{**}$ far from data (the subtracted term vanishes). That automatic "uncertainty grows away from data" behavior is the single most beautiful property of GP regression, and you get it for free.
>
> To *fit* the hyperparameters, we use the **log marginal likelihood**:
> $$ \log p(y \mid X) = -\tfrac{1}{2}\,y^\top (K_{XX} + \sigma^2 I)^{-1} y \;-\; \tfrac{1}{2}\log\lvert K_{XX} + \sigma^2 I\rvert \;-\; \tfrac{n}{2}\log 2\pi. $$
> The first term rewards fitting the data; the second, $-\tfrac12\log\lvert K + \sigma^2 I\rvert$, is an **automatic Occam penalty** that punishes overly flexible (wiggly, low-$\ell$) kernels. This built-in complexity control is why GPs resist overfitting *when the hyperpriors are sane*. The $(K + \sigma^2 I)^{-1}$ and the log-determinant are the $O(n^3)$ operations — Cholesky-factorizing an $n\times n$ matrix.

### 4.1 The `pm.gp.Marginal` API in three moves

PyMC's `pm.gp.Marginal` wraps all of that algebra. There are exactly three things you do with it:

1. **Construct** it with a covariance function (and optionally a mean function): `gp = pm.gp.Marginal(cov_func=cov_func)`.
2. **Fit** by attaching a marginal likelihood to your data: `gp.marginal_likelihood("y", X=X, y=y, sigma=sigma)`. This is the analytic $\mathcal{N}(\mathbf{m}_X, K_{XX} + \sigma^2 I)$ likelihood. Note the keyword is **`sigma=`** in current PyMC; older notebooks used `noise=` (the legacy spelling — both denote the additive observation noise, but write `sigma=`).
3. **Predict** at new inputs: `gp.conditional("f_pred", Xnew=Xnew)`. By default this returns the latent function $f$; pass `pred_noise=True` to get the predictive distribution **of observations** (i.e. $f$ + observation noise), which is what you want for honest prediction intervals on new $y$.

> ⚠️ **Pitfall:** `pm.gp.Marginal` has **no `.prior` method**. Calling `gp.prior(...)` on a Marginal raises `AttributeError`. The `Marginal` family uses `.marginal_likelihood` + `.conditional`; only `Latent`, `HSGP`, and `TP` expose `.prior`. Mixing these up is the second-most-common GP error after the `(n,1)` shape mistake.

---

## 5. Worked example: GP regression on synthetic data (recovering the truth)

There is no better way to *trust* a method than to generate data from a known process and watch the model recover it. Let me draw a single function from a GP with known $(\eta, \ell)$, add known observation noise $\sigma$, and check that the posterior finds all three. Synthetic-with-known-truth is the gold standard for understanding any estimator.

### 5.1 Generate the data

```python
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)

# Known truth: amplitude eta=1.5, lengthscale ell=1.0, observation noise sigma=0.3
n = 100
X = np.linspace(0, 10, n)[:, None]                      # (n, 1) — always 2-D
true_eta, true_ell, true_sigma = 1.5, 1.0, 0.3

# Build the true covariance and draw ONE function from it.
K_true = true_eta**2 * np.exp(-0.5 * (X - X.T)**2 / true_ell**2) + 1e-6 * np.eye(n)
f_true = rng.multivariate_normal(np.zeros(n), K_true)   # the latent function at X
y = f_true + rng.normal(0, true_sigma, size=n)          # noisy observations

plt.plot(X[:, 0], f_true, "k", lw=2, label="true f")
plt.plot(X[:, 0], y, "C0.", label="observed y")
plt.legend(); plt.title("Synthetic GP data with known eta=1.5, ell=1.0, sigma=0.3")
```

You will see a smooth black curve (the truth) with a scatter of blue dots around it (the noisy data). Our job: recover the black curve and the three numbers, with calibrated uncertainty, *seeing only the blue dots*.

### 5.2 Prior predictive: look before you leap

Before sampling, we sanity-check our priors by drawing functions from them. We standardize neither here (the data are already on a natural scale and centered near zero), but in real problems you would standardize $y$ and center $x$ — exactly the standardization-by-default discipline from Chapter 03. Our hyperpriors:

- $\ell \sim \text{InverseGamma}$ tuned so the prior excludes implausibly short lengthscales (the heart of §6, previewed here).
- $\eta \sim \text{HalfNormal}(\sigma=2)$ — weakly informative, allows amplitudes up to a few.
- $\sigma \sim \text{HalfNormal}(\sigma=1)$ — observation noise is positive and probably small.

```python
with pm.Model() as gp_model:
    # Lengthscale: InverseGamma kills ell ~ 0 and keeps a heavy right tail (see Section 6).
    ell = pm.InverseGamma("ell", alpha=4.0, beta=4.0)   # mean ~ beta/(alpha-1) = 1.33
    eta = pm.HalfNormal("eta", sigma=2.0)               # amplitude / vertical scale
    sigma = pm.HalfNormal("sigma", sigma=1.0)           # observation noise

    cov_func = eta**2 * pm.gp.cov.ExpQuad(input_dim=1, ls=ell)
    gp = pm.gp.Marginal(cov_func=cov_func)

    # Attach the marginal likelihood to the data. sigma= is the current keyword.
    y_ = gp.marginal_likelihood("y", X=X, y=y, sigma=sigma)
```

> 🩺 **Diagnostic — prior predictive functions.** You can visualize the prior over functions by sampling `ell`, `eta`, `sigma` from their priors, building $K_{XX}$ for each draw, and drawing a function from $\mathcal{N}(0, K_{XX})$ — exactly the loop from §1, but now with random hyperparameters. What you want to see: smooth-ish curves that wiggle on the *right order of magnitude* (a handful of bumps across the input range, not a hundred, not zero) and whose vertical spread plausibly contains your data. If the prior draws are all flat lines or all noise, fix your hyperpriors *now* — sampling will not rescue a bad prior.

### 5.3 Sample and diagnose

```python
with gp_model:
    idata = pm.sample(
        1000, tune=1000, chains=4, target_accept=0.9, random_seed=RANDOM_SEED,
    )

az.summary(idata, var_names=["ell", "eta", "sigma"])
```

A healthy run prints something like (your numbers will be close):

```
        mean    sd   hdi_3%  hdi_97%  ess_bulk  ess_tail  r_hat
ell     1.04   0.18    0.73     1.39      2100      2600   1.00
eta     1.42   0.27    0.95     1.93      1900      2400   1.00
sigma   0.29   0.03    0.24     0.34      3000      3100   1.00
```

Look at that: `ell ≈ 1.04` (truth 1.0), `eta ≈ 1.42` (truth 1.5), `sigma ≈ 0.29` (truth 0.3). The posterior recovered all three, and the true values sit comfortably inside the 94% HDIs. **R̂ = 1.00** for everything (well under our 1.01 target from Chapter 07), **ESS in the thousands** (far above the ≳400 floor), and you want **0 divergences**. Note that divergences are **not** a column in the `az.summary` table — check them separately with `idata.sample_stats.diverging.sum().item()` (you want `0`) or read the count PyMC prints at the end of sampling. That is what a clean GP fit looks like.

> 🩺 **Diagnostic — the trace and rank plots.** Run `az.plot_trace(idata, var_names=["ell", "eta", "sigma"])`. You want fuzzy-caterpillar traces with all four chains overlapping (good mixing) and unimodal marginal densities. Then `az.plot_rank(idata, var_names=["ell", "eta", "sigma"])` — rank plots should look like uniform histograms across chains; any chain that systematically sits high or low signals a mixing problem. If you instead see `ell` chains stuck at different values, suspect the lengthscale non-identifiability from §6.

> ⚠️ **Pitfall — divergences in GPs.** If you *do* see divergences here, the usual culprit is the lengthscale wandering toward zero into the flat, pathological region of the posterior. First response: confirm your `ell` prior actually penalizes short lengthscales (an InverseGamma or Gamma, **not** a HalfNormal or a flat prior). Second response: raise `target_accept` to 0.95 or 0.99. Third: standardize inputs and bump the Cholesky `jitter`. We unpack all of this in §6.

### 5.4 Predict with credible bands

Now the payoff — predictions at a dense grid of new inputs, including *beyond* the training range so you can watch the uncertainty fan out.

```python
Xnew = np.linspace(-2, 12, 300)[:, None]   # extends past the data on both sides

with gp_model:
    # Latent function f at new inputs (no observation noise):
    f_pred = gp.conditional("f_pred", Xnew=Xnew)
    # Predictive distribution of NEW observations (f + noise):
    y_pred = gp.conditional("y_pred", Xnew=Xnew, pred_noise=True)
    ppc = pm.sample_posterior_predictive(
        idata, var_names=["f_pred", "y_pred"], random_seed=RANDOM_SEED,
    )
    idata.extend(ppc)
```

To plot the function estimate with a 94% credible band:

```python
f_samples = idata.posterior_predictive["f_pred"]              # (chain, draw, Xnew)
f_mean = f_samples.mean(dim=["chain", "draw"]).values
# Pass the InferenceData GROUP (a Dataset) so we can select the variable by name,
# then index the 2-element `hdi` dimension explicitly (order-safe — no fragile .T unpack).
hdi = az.hdi(idata.posterior_predictive, var_names=["f_pred"], hdi_prob=0.94)["f_pred"]
f_lo = hdi.sel(hdi="lower").values
f_hi = hdi.sel(hdi="higher").values

plt.plot(X[:, 0], y, "C0.", alpha=0.6, label="data")
plt.plot(Xnew[:, 0], f_mean, "C1", lw=2, label="posterior mean f")
plt.fill_between(Xnew[:, 0], f_lo, f_hi, color="C1", alpha=0.25, label="94% HDI")
plt.legend(); plt.title("GP regression: posterior mean and uncertainty")
```

> 🩺 **Diagnostic — reading the credible band.** Three things to verify on this plot, every time. **(1)** Inside the data range, the orange mean curve should track the black truth (if you overplot it) and the band should be *narrow* — the GP is confident where it has data. **(2)** In the gaps and especially in the extrapolation regions ($x < 0$ and $x > 10$), the band should **fan out dramatically and the mean should revert toward the prior mean** (zero here) — the GP is honestly admitting ignorance. This fanning is the headline feature of GP uncertainty and is exactly what a polynomial or neural net does *not* give you for free. **(3)** The `y_pred` band (observations) is uniformly wider than the `f_pred` band (latent function) by roughly the observation noise $\sigma$ — because predicting a new noisy measurement is harder than predicting the underlying signal.

---

## 6. Priors on the lengthscale — Betancourt's robust-GP argument

This is the most important section in the chapter for anyone who wants to *fit* GPs rather than just admire them. I told you the lengthscale/noise tug-of-war was the central difficulty; here is how we resolve it, and *why* a careless prior leads to an hour of sampling and a screen full of divergences.

### 6.1 Why a too-short lengthscale is a disaster — twice over

**Statistically**, a too-short lengthscale overfits. With $\ell$ tiny, the GP decorrelates over a sub-data-spacing distance, so it can place an independent little bump at every observation and thread the curve through all the noise. Training fit is perfect; generalization is garbage; uncertainty bands are absurdly tight where they should be wide.

**Geometrically — and this is Betancourt's deep point — the trouble is worse than overfitting.** With finite data, the likelihood is **essentially flat as $\ell \to 0$**: many tiny lengthscales fit the data about equally well, because once $\ell$ is smaller than the gap between data points, shrinking it further changes almost nothing the data can see. A flat likelihood means a non-identified parameter, and a non-identified parameter means a **degenerate posterior geometry** — long flat ridges and funnels that Hamiltonian Monte Carlo cannot traverse. The symptom is divergences, low ESS, and chains that wander off toward $\ell = 0$ and get stuck. The data *cannot* tell the sampler where to stop, so *you* must, via the prior.

> 📜 **Citation/Origin:** This is the core message of Betancourt's case study *"Robust Gaussian Process Modeling"* (betanalpha.github.io). The fix — a prior that puts essentially **zero density at $\ell = 0$** and a **heavy right tail** — appears there and in the Stan User's Guide's GP chapter, and PyMC's own GP examples follow it.

### 6.2 The InverseGamma prescription

The **InverseGamma** distribution is the standard choice because it has exactly the two properties we need:

- **Zero density at zero.** $\text{InverseGamma}$ has $p(\ell) \to 0$ as $\ell \to 0$ (in fact exponentially fast). This *kills* the pathological short-lengthscale region before HMC can fall into it.
- **A heavy right tail.** It still allows large, smooth lengthscales if the data genuinely support them — you are not forcing wiggliness, just forbidding the absurd.

Contrast with the alternatives: a flat or uniform prior gives the flat likelihood free rein (disaster); a HalfNormal has *nonzero* density at zero and a light tail (lets $\ell$ collapse and over-penalizes large $\ell$); a Gamma is a perfectly good practical alternative that also pushes mass off zero (and is what the Mauna Loa example uses). InverseGamma is the textbook default.

> 🔧 **In practice — how to set the two parameters.** Betancourt's recipe is a tail-calibration: choose the InverseGamma so that
> - about **1% of the prior mass falls below** the smallest lengthscale you could possibly resolve — roughly the *typical spacing between your data points* (you cannot learn structure finer than your sampling), and
> - about **1% falls above** the *range of your input domain* (you cannot learn structure longer than the window you observed).
>
> In other words, concentrate the prior on the band of lengthscales that are *physically learnable from this dataset*. PyMC lets you parameterize by mean and standard deviation directly, `pm.InverseGamma("ell", mu=..., sigma=...)`, which is often easier than thinking in $(\alpha, \beta)$. If you have PreliZ installed, `pz.maxent(pz.InverseGamma(), lower, upper, mass=0.98)` will solve the parameters from the interval $[\text{lower}, \text{upper}]$ for you — a clean way to encode "98% of the prior lies between the min spacing and the domain width." Otherwise, solve the two tail-probability constraints numerically, or just pick $(\alpha, \beta)$ by plotting `pm.draw(pm.InverseGamma.dist(alpha=a, beta=b), 5000)` and eyeballing the histogram against your learnable band.

> 🧮 **The math — a concrete tuning.** Suppose your inputs span $[0, 10]$ with $n=100$ points, so the spacing is $\approx 0.1$ and the domain width is $10$. The strict Betancourt calibration asks for $\sim$1% mass below $\approx 0.1$ and $\sim$1% above $\approx 10$. The `InverseGamma(alpha=4, beta=4)` we used in §5.2 does **not** hit that target exactly — it is *tighter* than the strict recipe. Its mean is $\beta/(\alpha-1) \approx 1.33$, and numerically it puts about **98% of its mass in roughly $[0.40, 4.86]$** (those are its 1%/99% quantiles), with only $\approx 5\times10^{-14}$ mass below $0.1$ and $\approx 0.08\%$ mass above $10$. So this prior is *more concentrated* than "1% below min-spacing, 1% above domain width": it sits comfortably off zero and squarely inside the learnable band for this $n=100$, range-10 problem — which is why the §5 fit was clean — but it is not the literal tail calibration. If you want the strict two-tail constraint, solve for $(\alpha, \beta)$ from the interval directly: with PreliZ, `pz.maxent(pz.InverseGamma(), lower=0.1, upper=10, mass=0.98)` returns parameters whose 1%/99% quantiles are exactly $0.1$ and $10$ (a much wider, heavier-tailed prior than `(4, 4)`). Always re-tune to your own spacing and domain; do not copy numbers blindly.

> ⚠️ **Pitfall:** Do not slap a `HalfNormal` or `HalfCauchy` on the lengthscale "because we use them for scale parameters everywhere else." Those have positive density at zero and will reintroduce the funnel. The lengthscale is special: it needs a prior that is *zero at zero*. This is one of the few places in the course where the standard scale-prior advice (Chapter 03) does **not** apply.

### 6.3 Priors on amplitude and noise

These are easier. The **amplitude** $\eta$ is an ordinary positive scale parameter, so the usual weakly-informative choices apply: `pm.HalfNormal("eta", sigma=2)` or `pm.Exponential("eta", lam=1)` are both fine — set the scale to a few times the standard deviation of your (standardized) outcome. The **noise** $\sigma$ is likewise a positive scale: `pm.HalfNormal("sigma", sigma=1)` on standardized data is a sensible default. The one subtlety: $\eta$ and $\sigma$ can trade off (signal variance vs noise variance both explain scatter), so give $\sigma$ a prior that mildly favors *some* noise rather than zero — an `Exponential` or `HalfNormal` does this naturally, whereas a prior that allows $\sigma \to 0$ invites the interpolate-everything solution.

---

## 7. Latent GPs for non-Gaussian likelihoods — `pm.gp.Latent`

The `Marginal` trick — integrate out $f$ analytically — works **only** because both the GP prior and the likelihood are Gaussian. The moment your outcome is binary, a count, or anything non-Gaussian, the conjugacy breaks and the integral has no closed form. Now you must keep the latent function values $f = (f(x_1), \dots, f(x_n))$ as *explicit parameters* and let MCMC explore them jointly with the hyperparameters. That is what `pm.gp.Latent` does.

The generative story for, say, **GP classification**:

$$
f \sim \mathcal{GP}(0, k), \qquad p_i = \text{logit}^{-1}\!\big(f(x_i)\big), \qquad y_i \sim \text{Bernoulli}(p_i).
$$

The GP gives you a smooth latent function over the input space; the inverse-logit squashes it into $[0,1]$ probabilities; the Bernoulli generates the labels. For **Poisson counts** you would use a log link instead: $\lambda_i = \exp(f(x_i))$, $y_i \sim \text{Poisson}(\lambda_i)$. The pattern is identical to the GLMs of Chapter 09 — a link function and an outcome distribution — except the linear predictor is replaced by a flexible GP-distributed function.

> 🧮 **The math — why latent GPs are hard, and the reparameterization that saves them.** With $n$ data points you now have $n$ correlated latent values $f$ plus the hyperparameters, and those $f$ are *strongly* correlated with each other (that is the whole point of the kernel) and with the lengthscale. Sampling a high-dimensional, strongly-correlated Gaussian directly is exactly the kind of geometry HMC struggles with. PyMC handles this automatically by sampling in a **whitened / non-centered** parameterization: instead of sampling $f$ directly, it samples i.i.d. standard normals $v \sim \mathcal{N}(0, I)$ and sets $f = L v$, where $L$ is the Cholesky factor of $K_{XX}$. The $v$'s are uncorrelated and unit-scale — easy for HMC — and the correlation structure is reintroduced deterministically by $L$. This is the *same* non-centered idea you met as the cure for the funnel in Chapters 05, 07, and 10; here PyMC applies it for you under the hood (`reparameterize=True` by default). **It is the only reason latent GPs sample at all.**

### 7.1 The `pm.gp.Latent` API

`Latent` uses `.prior` + `.conditional` (it has **no** `.marginal_likelihood` — the mirror image of `Marginal`):

- **Construct:** `gp = pm.gp.Latent(cov_func=cov)`.
- **Place the latent function:** `f = gp.prior("f", X=X)`. This `f` is a PyMC random variable holding the latent function values at the training inputs. You then feed `f` through your link and likelihood.
- **Predict:** `f_pred = gp.conditional("f_pred", Xnew)`.

### 7.2 Worked example: GP classification

A small synthetic two-class problem where the decision boundary moves smoothly across the input.

```python
import numpy as np
import pymc as pm
import arviz as az

RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)

# Inputs and a smooth latent logit that the GP should recover.
n = 80
x = np.linspace(0, 10, n)
X = x[:, None]                                  # (n, 1)
true_f = 2.0 * np.sin(x) - 0.5 * x + 2.0        # the latent function (unknown to the model)
p = 1.0 / (1.0 + np.exp(-true_f))               # true probabilities
y = rng.binomial(1, p)                          # observed 0/1 labels

with pm.Model() as latent_model:
    # InverseGamma lengthscale (mu/sigma parameterization), same robustness logic as Section 6.
    ell = pm.InverseGamma("ell", mu=2.0, sigma=1.0)
    eta = pm.Exponential("eta", lam=1.0)                 # amplitude
    cov = eta**2 * pm.gp.cov.ExpQuad(input_dim=1, ls=ell)

    gp = pm.gp.Latent(cov_func=cov)
    f = gp.prior("f", X=X)                               # latent function values at X

    p_ = pm.Deterministic("p", pm.math.invlogit(f))      # squash to probabilities
    y_ = pm.Bernoulli("y", p=p_, observed=y)             # Bernoulli likelihood

    idata = pm.sample(
        1000, tune=1000, chains=4, target_accept=0.95,   # latent GPs like a higher target
        random_seed=RANDOM_SEED,
    )
```

> 🔧 **In practice:** Notice `target_accept=0.95` rather than 0.9. Latent GPs have trickier geometry than the marginal version (you are sampling all those correlated $f$'s), so a higher target step-size acceptance — meaning smaller, more careful steps — usually pays for itself in avoided divergences. If you still see divergences, go to 0.99 and double-check the lengthscale prior.

Predicting the latent function (and hence the probability) at new inputs:

```python
Xnew = np.linspace(-1, 11, 200)[:, None]
with latent_model:
    f_pred = gp.conditional("f_pred", Xnew, jitter=1e-4)   # bump jitter if Cholesky complains
    ppc = pm.sample_posterior_predictive(
        idata, var_names=["f_pred"], random_seed=RANDOM_SEED,
    )
    idata.extend(ppc)

# Convert latent f draws to probability draws, then summarize.
# (numpy is already imported at the top of this example; no need to re-import.)
f_draws = idata.posterior_predictive["f_pred"]
p_draws = 1.0 / (1.0 + np.exp(-f_draws))                    # still a DataArray named "f_pred"
p_mean = p_draws.mean(dim=["chain", "draw"]).values
# az.hdi on a bare DataArray returns a DataArray with a length-2 `hdi` dim
# (coords "lower"/"higher"); select it explicitly. Do NOT subscript ["f_pred"] —
# the result is a DataArray, not a Dataset, so that would raise.
p_hdi = az.hdi(p_draws, hdi_prob=0.94)
p_lo = p_hdi.sel(hdi="lower").values
p_hi = p_hdi.sel(hdi="higher").values
```

> 🩺 **Diagnostic — what the classification fit shows.** Plot `p_mean` against `Xnew` with the 94% band, and overlay the observed 0/1 labels as a rug. You should see a smooth probability curve that rises toward 1 where the data are mostly 1's and falls toward 0 where they are mostly 0's, with the band *widening* in regions with few observations and at the extrapolation edges — once again, honest uncertainty. Compare `p_mean` to the true `p`: the GP should track it without anywhere committing to a hard 0/1 it cannot justify. That graceful, uncertainty-aware decision boundary is exactly what logistic regression with a fixed linear predictor cannot give you when the truth is nonlinear.

> ⚠️ **Pitfall — Cholesky failures in `.conditional`.** If `gp.conditional` throws `LinAlgError: ... not positive definite`, the prediction covariance has become numerically singular — usually from near-duplicate `Xnew` points or a very small lengthscale. The fix is to **raise the `jitter`** (e.g. `jitter=1e-4` as shown, or `1e-3`), which adds a tiny ridge to the diagonal to restore positive-definiteness. Standardizing inputs and de-duplicating `Xnew` also help.

---

## 8. The $O(n^3)$ wall, and how to climb it

Everything above is exact — and exactly $O(n^3)$ in time and $O(n^2)$ in memory, because of that $n\times n$ Cholesky. At $n=1{,}000$ you wait seconds. At $n=10{,}000$ you wait minutes per gradient evaluation and your matrix is 800 MB. At $n=100{,}000$ a dense GP is simply impossible on a laptop. Since every NUTS step needs gradients of the log marginal likelihood, this cubic cost is multiplied by thousands of leapfrog steps. So scaling GPs is not a luxury — it is the difference between a toy and a tool.

There are two production-grade escape hatches in PyMC. They attack the cost from different angles.

### 8.1 HSGP — the Hilbert-space basis approximation (the modern default)

The **Hilbert-Space Gaussian Process (HSGP)** is, in my opinion, the single most useful GP technique for practitioners working in one or a few input dimensions. The idea is gorgeous.

> 🧮 **The math — the HSGP approximation.** Any stationary kernel has a **spectral density** $S(\omega)$ (its Fourier transform — Bochner's theorem). HSGP approximates the GP by expanding the function in a fixed set of $m$ **Laplacian eigenfunctions** $\phi_j(x)$ on a bounded domain $[-L, L]$, with the kernel's spectral density supplying the coefficients' prior variances:
> $$ f(x) \approx \sum_{j=1}^{m} \sqrt{S\big(\sqrt{\lambda_j}\big)}\;\phi_j(x)\;\beta_j, \qquad \beta_j \sim \mathcal{N}(0, 1). $$
> Here $\lambda_j$ and $\phi_j$ are the (known, closed-form) eigenvalues and eigenfunctions of the Laplacian on $[-L, L]$. The crucial property: **the basis functions $\phi_j$ do not depend on the kernel hyperparameters** $(\ell, \eta)$. Only the coefficient variances $S(\sqrt{\lambda_j})$ depend on them. So you compute the $n \times m$ basis matrix **once**, up front, and during sampling you only ever do a cheap matrix–vector product. The cost collapses from $O(n^3)$ to roughly **$O(nm + m)$**. A GP becomes a linear regression on a clever fixed basis.

> 📜 **Citation/Origin:** Solin & Särkkä (2019, *Statistics and Computing*) derived the reduced-rank Hilbert-space method; Riutort-Mayol, Bürkner, Andersen, Solin & Vehtari (2022, arXiv:2004.11408) worked out the practical $m$/$c$/$L$ choices for probabilistic programming, which is what PyMC's `pm.gp.HSGP` implements.

The `pm.gp.HSGP` API mirrors `Latent` (it is a latent representation): `.prior` + `.conditional`.

```python
import numpy as np, pymc as pm

# Reusing the synthetic regression data from Section 5 (X, y), now via HSGP.
with pm.Model() as hsgp_model:
    ell = pm.InverseGamma("ell", alpha=4.0, beta=4.0)
    eta = pm.HalfNormal("eta", sigma=2.0)
    cov_func = eta**2 * pm.gp.cov.ExpQuad(input_dim=1, ls=ell)

    # m = number of basis functions per dim; c = boundary extension factor.
    # c=2 keeps the boundary comfortably outside the data (the chapter's "often 2-4"
    # guidance); c=1.2 is the documented floor and can bias the fit near the edges.
    gp = pm.gp.HSGP(m=[200], c=2.0, cov_func=cov_func)
    f = gp.prior("f", X=X)                       # HSGP uses .prior, like Latent

    sigma = pm.HalfNormal("sigma", sigma=1.0)
    y_ = pm.Normal("y", mu=f, sigma=sigma, observed=y)   # explicit Gaussian likelihood on f

    idata_hsgp = pm.sample(
        1000, tune=1000, chains=4, target_accept=0.9, random_seed=RANDOM_SEED,
    )
```

Notice the structural difference from `pm.gp.Marginal`: HSGP gives you the **latent function `f` explicitly** (via `.prior`), so you write the likelihood yourself — here `pm.Normal("y", mu=f, sigma=sigma, observed=y)`. That same flexibility means HSGP works with *any* likelihood (Bernoulli, Poisson, …), not just Gaussian. It is a latent-GP accelerator.

**Choosing `m` and `c`** — the two knobs that control approximation quality:

- **`m`** (a list, one entry per active input dimension) is the number of basis functions. More basis functions can represent **shorter** lengthscales / finer wiggles, at higher cost. In multiple dimensions the basis is a tensor product: `m=[25, 25]` ⇒ $625$ basis functions. If your HSGP fit looks too smooth and misses real wiggles, **increase `m`**.
- **`c`** is the boundary extension factor. Internally the domain is $[-L, L]$ with $L = c \cdot S$ where $S$ is the half-width of your (centered) inputs. Larger `c` lets the approximation represent **longer** lengthscales, but generally needs a larger `m` to keep short-scale accuracy. Riutort-Mayol et al. recommend **`c ≥ 1.2`** at minimum, and often $2$–$4$. Critically, **all your data — training *and* prediction inputs — must sit well inside $[-L, L]$**, or the approximation degrades badly at the edges.

> 🔧 **In practice — let PyMC size them for you.** PyMC ships a helper, `pm.gp.hsgp_approx.approx_hsgp_hyperparams(x_range, lengthscale_range, cov_func)`, that returns recommended `(m, c)` given the span of your inputs and the *range of lengthscales you expect to need*. For a wide domain spanning many lengthscales (e.g. `x_range=[-5, 95], lengthscale_range=[1, 50], cov_func="matern52"`), it returns recommendations on the order of a few hundred basis functions and a `c` around `4` — run it yourself to get the exact values for your problem rather than memorizing a number. The supported `cov_func` strings are `"expquad"`, `"matern52"`, `"matern32"`. Use this rather than guessing. *(# verify exact return values in current PyMC docs.)*

To predict on a new grid — the same `.prior` + `.conditional` pattern as `Latent` — make sure every prediction point sits **well inside** the boundary $[-L, L]$ that `c` set:

```python
Xnew = np.linspace(0, 10, 300)[:, None]          # stay inside the boundary (no big extrapolation)
with hsgp_model:
    f_pred = gp.conditional("f_pred", Xnew=Xnew)             # HSGP uses .conditional, like Latent
    idata_hsgp.extend(
        pm.sample_posterior_predictive(idata_hsgp, var_names=["f_pred"], random_seed=RANDOM_SEED)
    )
```

> 🩺 **Diagnostic — did the approximation hold?** After fitting, compare the HSGP posterior summaries for `ell`, `eta`, `sigma` against the exact `Marginal` fit from §5 on the same data. With enough basis functions and the boundary set comfortably outside the data, they should land close to the exact fit (which recovered roughly `ell ≈ 1.0`, `eta ≈ 1.4`, `sigma ≈ 0.3`) — expect agreement to within Monte Carlo error, not to the last decimal. If HSGP's `ell` comes out systematically larger than the exact fit's, you probably have too few basis functions (`m` too small) to capture the true wiggliness, so the model compensates with a longer lengthscale — bump `m` and refit. If the fit looks biased only near the edges of the data, your boundary is too tight: raise `c` (we used `c=2`; push toward `3`–`4` if needed). PyMC's HSGP-Basic example also shows a "Gram-matrix validation" where you compare the approximate kernel matrix $\Phi \Lambda \Phi^\top$ to the exact $K_{XX}$ — a direct check that your `m`/`c` are adequate.

**The speedup, concretely.** On the §5 synthetic data ($n=100$) the exact `Marginal` already runs fast, but the scaling story is what matters. The exact GP cost grows as $n^3$; HSGP grows as $n \cdot m$. At $n=100$ with $m=200$ they are comparable, but at $n=10{,}000$ the exact GP needs $\sim 10^{12}$ operations per gradient while HSGP with $m=200$ needs $\sim 2\times 10^6$ — a *six-order-of-magnitude* difference. In wall-clock terms, a model that would take hours (or be infeasible) finishes in seconds. **This is why HSGP, not the exact GP, is what you reach for on any real-sized 1-D or low-D problem.**

> 🔧 **In practice — periodic HSGP.** Plain `HSGP` supports `ExpQuad`, `Matern52`, and `Matern32` (kernels with a known spectral density), but **not** arbitrary kernels and **not** `Periodic` directly. For a fast periodic component there is a separate class, **`pm.gp.HSGPPeriodic`**, built around a `Periodic` covariance. *(# verify HSGPPeriodic constructor args — m, scale, cov_func — in current PyMC docs.)*

### 8.2 Sparse / inducing-point GPs — `pm.gp.MarginalApprox`

The other classic approach keeps the GP machinery but summarizes the data through a small set of **$m$ inducing points** $X_u$ — pseudo-observations that act as a low-rank bottleneck. Instead of an $n \times n$ matrix you work with $n \times m$ and $m \times m$ matrices, dropping the cost to **$O(n m^2)$**.

```python
import pymc as pm

# Choose inducing points: k-means cluster centers of X are a standard heuristic.
Xu = pm.gp.util.kmeans_inducing_points(20, X)    # 20 inducing points
# (# verify pm.gp.util.kmeans_inducing_points arg order in current PyMC docs)

with pm.Model() as sparse_model:
    ell = pm.InverseGamma("ell", alpha=4.0, beta=4.0)
    eta = pm.HalfNormal("eta", sigma=2.0)
    sigma = pm.HalfNormal("sigma", sigma=1.0)
    cov_func = eta**2 * pm.gp.cov.ExpQuad(input_dim=1, ls=ell)

    gp = pm.gp.MarginalApprox(cov_func=cov_func, approx="VFE")   # VFE / FITC / DTC
    y_ = gp.marginal_likelihood("y", X=X, Xu=Xu, y=y, sigma=sigma)
    idata_sparse = pm.sample(
        1000, tune=1000, chains=4, target_accept=0.9, random_seed=RANDOM_SEED,
    )
```

`MarginalApprox` is the marginal (Gaussian-likelihood) sparse approximation — it keeps `.marginal_likelihood` + `.conditional`, like `Marginal`, but takes the inducing points `Xu`. Three approximation flavors:

- **`"VFE"`** (Variational Free Energy; Titsias 2009) — the **default and usually safest**. It is a proper variational lower bound and does not pretend to know more than it does.
- **`"FITC"`** (Fully Independent Training Conditional) — can be more flexible but is known to *underestimate noise* and sometimes overfit.
- **`"DTC"`** (Deterministic Training Conditional) — the simplest, least recommended.

> ⚠️ **Pitfall — renamed class.** In PyMC v3 this class was called `MarginalSparse`. In v5 it is **`MarginalApprox`**. If you find old code or blog posts using `pm.gp.MarginalSparse`, that is the legacy name; use `MarginalApprox`. Also note: the sparse marginal approximation assumes **white i.i.d. noise**, so `sigma` here must be a scalar, not a Covariance.

> 💡 **Intuition — HSGP vs sparse, which to use.** Both cut the cubic cost; they trade off differently. **HSGP** shines in **low dimensions** (1–3 inputs) and is *blazingly* fast because the basis is hyperparameter-independent — it is my default for time series and 1-D/2-D smoothing. **Inducing-point sparse GPs** scale better to **higher input dimensions** and to genuinely large $n$ where you just need a coarse summary, and they extend naturally to non-Gaussian likelihoods via their latent cousins. For most readers of this course working with 1-D or 2-D problems, **HSGP is the first thing to try**; reach for `MarginalApprox` when dimensionality climbs.

| | Exact (`Marginal`/`Latent`) | HSGP | Sparse (`MarginalApprox`) |
|---|---|---|---|
| Cost per gradient | $O(n^3)$ | $O(nm)$ | $O(nm^2)$ |
| Best input dimension | any (but small $n$) | low (1–3) | low–moderate |
| Non-Gaussian likelihoods | `Latent` | yes (it's latent) | via latent variants |
| Main knob | — | $m$, $c$ | #inducing points $m$, `approx` |
| When to use | $n \lesssim 1000$, need exactness | 1-D/2-D, large $n$ | higher-D, large $n$ |

---

## 9. The showcase: decomposing Mauna Loa CO₂

Now we put every idea together on the most famous dataset in the GP world: the **Mauna Loa atmospheric CO₂** record. Since 1958, the observatory on Mauna Loa, Hawaii has measured atmospheric CO₂. The series has three structures stacked on top of each other, and a GP with an **additive kernel** can pull them apart cleanly:

1. a **long, smooth rising trend** (the secular increase in CO₂),
2. a **seasonal cycle** of period 1 year (plants breathe in summer, exhale in winter), whose shape may drift slowly over decades,
3. **short-scale wiggles and noise** (medium-term irregularities plus measurement scatter).

The modeling philosophy is the heart of this chapter: **each structure gets its own kernel, and we add them.** The sum kernel says "the function is a sum of independent components." Because the GP is additive, we can later predict each component *separately* — literally decompose the data into trend, season, and noise, with uncertainty on each.

### 9.1 Load and prepare the data

```python
import statsmodels.api as sm
import numpy as np
import pandas as pd

co2 = sm.datasets.co2.load_pandas().data          # weekly CO2 ppm, 1958–2001
co2 = co2.dropna().resample("MS").mean().dropna()  # monthly means

# x = years since start (a natural, centered-ish input); y = CO2 in ppm.
t = (co2.index - co2.index[0]).days / 365.25
x = t.values[:, None]                              # (n, 1)
y_raw = co2["co2"].values

# Standardize y so amplitude/noise priors are on a sane scale (standardization-by-default).
y_mean, y_std = y_raw.mean(), y_raw.std()
y = (y_raw - y_mean) / y_std
```

> 🔧 **In practice:** We standardize $y$ (and keep $x$ as years-since-start, which is already centered near the data) precisely so the amplitude and noise priors — `HalfNormal(sigma=2)`-ish — are sensible without per-dataset re-tuning. This is the standardization discipline from Chapter 03 paying off again. Remember to *un-standardize* predictions at the end by multiplying by `y_std` and adding `y_mean`.

### 9.2 Design the additive kernel

This is where you *think like a modeler*. Each kernel encodes a hypothesis about a component:

- **Trend** — a long-lengthscale smooth kernel. Either an `ExpQuad` with a large lengthscale, or a `Matern52`. It captures the slow rise.
- **Seasonal** — `Periodic(period=1.0)` (one-year cycle) **multiplied by** an `ExpQuad` with a long lengthscale. The product is the key trick: the periodic part repeats yearly, and the ExpQuad envelope lets the *shape* of the seasonal cycle drift slowly across the decades instead of repeating identically forever. A pure `Periodic` would force every year's curve to be byte-for-byte identical, which is too rigid.
- **Medium-term irregularities** — a `RatQuad` (rational quadratic), which is a scale-mixture of ExpQuads and so naturally captures variation across a *range* of medium lengthscales.
- **Noise** — a short-lengthscale `Matern32` plus an explicit `WhiteNoise`/`sigma` for the i.i.d. measurement scatter.

```python
import pymc as pm

with pm.Model() as co2_model:
    # --- Long smooth trend ---
    eta_trend = pm.HalfNormal("eta_trend", sigma=2.0)
    ell_trend = pm.Gamma("ell_trend", alpha=4.0, beta=0.1)      # large lengthscale (years)
    cov_trend = eta_trend**2 * pm.gp.cov.ExpQuad(input_dim=1, ls=ell_trend)

    # --- Decaying seasonal: Periodic (1 yr) * ExpQuad envelope ---
    eta_seas = pm.HalfNormal("eta_seas", sigma=1.0)
    ell_seas = pm.Gamma("ell_seas", alpha=4.0, beta=2.0)         # within-year wiggliness
    ell_decay = pm.Gamma("ell_decay", alpha=4.0, beta=0.05)      # how slowly the season drifts
    cov_seasonal = (
        eta_seas**2
        * pm.gp.cov.Periodic(input_dim=1, period=1.0, ls=ell_seas)
        * pm.gp.cov.ExpQuad(input_dim=1, ls=ell_decay)
    )

    # --- Medium-term irregularities (rational quadratic = mixture of lengthscales) ---
    eta_med = pm.HalfNormal("eta_med", sigma=1.0)
    ell_med = pm.Gamma("ell_med", alpha=4.0, beta=1.0)
    alpha_med = pm.Gamma("alpha_med", alpha=2.0, beta=1.0)
    cov_medium = eta_med**2 * pm.gp.cov.RatQuad(
        input_dim=1, ls=ell_med, alpha=alpha_med
    )

    # --- Short-scale noise component ---
    eta_noise = pm.HalfNormal("eta_noise", sigma=0.5)
    ell_noise = pm.Gamma("ell_noise", alpha=2.0, beta=4.0)       # very short lengthscale
    cov_noise = eta_noise**2 * pm.gp.cov.Matern32(input_dim=1, ls=ell_noise)

    # --- Build component GPs and ADD them ---
    gp_trend    = pm.gp.Marginal(cov_func=cov_trend)
    gp_seasonal = pm.gp.Marginal(cov_func=cov_seasonal)
    gp_medium   = pm.gp.Marginal(cov_func=cov_medium)
    gp_noise    = pm.gp.Marginal(cov_func=cov_noise)
    gp = gp_trend + gp_seasonal + gp_medium + gp_noise           # additive GP

    # --- White i.i.d. measurement noise via sigma ---
    sigma = pm.HalfNormal("sigma", sigma=0.5)
    y_ = gp.marginal_likelihood("y", X=x, y=y, sigma=sigma)

    idata_co2 = pm.sample(
        1000, tune=1000, chains=4, target_accept=0.9, random_seed=RANDOM_SEED,
    )
```

> 🔧 **In practice — why Gamma instead of InverseGamma here.** The Mauna Loa example uses `Gamma` lengthscale priors. Gamma also has zero density at zero (so it tames the short-lengthscale pathology) and is easy to shape by mean/variance for each component — a long-trend `ell_trend` wants a prior centered on *tens of years*, while `ell_noise` wants one centered on *months*. Both Gamma and InverseGamma are legitimate robust choices; the rule that matters is **zero density at zero plus a controlled tail**, tuned to each component's expected scale. Set each lengthscale prior to sit near the physical scale of the structure it models.

> ⚠️ **Pitfall — additive GPs only add within the same class.** You can only add GPs of the **same class** (`Marginal + Marginal`, or `Latent + Latent`), and the addition is on the GP *objects*, not the covariance objects. (You *can* also just add the covariance functions and build one GP — both work; adding GP objects is what lets you predict components separately, next.)

### 9.3 Diagnose

```python
import arviz as az
az.summary(idata_co2, var_names=["ell_trend", "ell_seas", "eta_trend", "eta_seas", "sigma"])
```

> 🩺 **Diagnostic.** Check the usual suspects from Chapter 07: **R̂ ≤ 1.01** on every hyperparameter, **ESS_bulk and ESS_tail ≳ 400**, **0 divergences**. Additive multi-kernel GPs have *many* correlated hyperparameters (each component has its own $\eta$ and $\ell$, and the amplitudes can trade off against each other), so do not be surprised if a few parameters are weakly identified and have wider posteriors — that is fine as long as R̂ and ESS are healthy and divergences are zero. If you see divergences, the prime suspect is again a lengthscale collapsing; tighten the relevant Gamma prior and/or raise `target_accept` to 0.95.

### 9.4 Predict — and decompose into components

This is the magic moment. Because the GP is a *sum* of component GPs, we can ask each component to predict on its own, using the `given=` argument so the sub-component knows the full data context:

```python
x_future = np.linspace(x.min(), x.max() + 10, 600)[:, None]   # extend 10 years past the data

# Full-model context that every sub-component needs:
given = {"X": x, "y": y, "sigma": sigma, "gp": gp}

with co2_model:
    f_full  = gp.conditional("f_full",   x_future)                       # everything
    f_trend = gp_trend.conditional("f_trend", x_future, given=given)     # trend ONLY
    f_seas  = gp_seasonal.conditional("f_seas", x_future, given=given)   # seasonal ONLY
    ppc = pm.sample_posterior_predictive(
        idata_co2, var_names=["f_full", "f_trend", "f_seas"],
        random_seed=RANDOM_SEED,
    )
    idata_co2.extend(ppc)
```

> ⚠️ **Pitfall — the mandatory `given=` dict.** When you predict a *sub-component* of an additive GP, you **must** pass `given={"X": X, "y": y, "sigma": sigma, "gp": gp_full}`, where `gp_full` is the complete summed GP. Forget it and the sub-component has no idea about the data or the other components, and you get nonsense predictions. The full-model `gp.conditional` does not need `given=` because it already *is* the full context.

> 🩺 **Diagnostic — reading the decomposition.** Plot the three predicted components (remember to multiply back by `y_std` to return to ppm). You should see: **(1) the trend** — a smooth, relentlessly rising curve, with its uncertainty band widening in the 10-year extrapolation but the *direction* of the rise clearly maintained; **(2) the seasonal component** — a clean ~1-year oscillation of a few ppm, whose envelope is allowed to slowly grow over the decades (the product kernel at work); **(3) the full prediction** — trend + season + medium-term, tracking the sawtooth-on-a-ramp shape of the raw data tightly inside the observed window and fanning into honest uncertainty beyond it. *This decomposition — pulling a messy series apart into interpretable, individually-uncertain pieces — is the GP at its most powerful, and it is exactly the structural-time-series view we return to in Chapter 15.*

---

## 10. Connections: GPs ↔ splines ↔ BART ↔ infinite neural nets

A GP does not live in isolation. Knowing its neighbors tells you *when to reach for it* versus something else — the central judgment call of Chapter 14.

**GP ↔ splines.** A **smoothing spline** — penalized regression on a basis with a roughness penalty — is *exactly* the posterior mean of a GP with a particular kernel. This is the Kimeldorf–Wahba theorem (1970): the function that minimizes "fit error + roughness penalty" coincides with the GP posterior mean for the matching covariance. The HSGP approximation makes the connection vivid and computational: an HSGP literally *is* a basis-function regression (eigenfunctions of the Laplacian) with a principled prior on the coefficients. So a spline is a GP with the Bayesian uncertainty quantification stripped off, and a GP is a spline that knows how unsure it is.

> 📜 **Citation/Origin:** Kimeldorf & Wahba (1970); see also Rasmussen & Williams Ch. 6.3 for the spline–GP correspondence, and Wahba's *Spline Models for Observational Data* (1990).

**GP ↔ BART.** BART (Bayesian Additive Regression Trees, Chapter 14) is the GP's tree-based cousin: also a flexible nonparametric regression with a regularizing prior, also giving full posterior uncertainty. The differences are what matter. A stationary GP assumes the *same* smoothness everywhere (one global lengthscale); BART is **adaptive and non-stationary** — it can be flat in one region and jagged in another, and it handles **step changes, many features, and interactions** far more gracefully than a stationary GP. The price is that BART's uncertainty is rougher and less smoothly calibrated than a GP's. Rule of thumb: **smooth, low-dimensional function with beautifully calibrated uncertainty ⇒ GP; many features / interactions / step-like or discontinuous structure ⇒ BART; a known basis and a single 1-D wiggle ⇒ a spline.**

**GP ↔ infinite-width neural networks.** Here is the one that surprises everyone. Neal (1996) proved that a single-hidden-layer neural network with i.i.d. random weights, in the limit of **infinitely many hidden units**, converges to a Gaussian process — the kernel determined by the activation function and the weight priors. The modern extension (the *Neural Tangent Kernel*, Jacot et al. 2018, and the *deep NN-GP* correspondence, Lee et al. 2018) shows that infinitely-wide deep networks are *also* GPs. So the GP is, in a precise sense, the infinite-width limit of a Bayesian neural network. This is why GPs are often called the "non-parametric Bayesian" workhorse and why intuition flows both ways: a GP is what a neural net *wants to be* when it has infinitely many neurons and you integrate out the weights.

> 📜 **Citation/Origin:** Neal, *Bayesian Learning for Neural Networks* (1996), Ch. 2; Lee et al., "Deep Neural Networks as Gaussian Processes" (ICLR 2018); Jacot, Gabriel & Hongler, "Neural Tangent Kernel" (NeurIPS 2018).

> 💡 **Intuition — the unifying view.** Splines, GPs, BART, and wide neural nets are all answers to the same question: *how do I learn a flexible function with a controlled amount of wiggliness?* They differ in **how** they parameterize flexibility (basis + penalty, kernel, trees, infinite layers) and in **how honestly** they report uncertainty. The GP is the one with the cleanest probabilistic semantics — which is exactly why it is worth the cubic cost (or the HSGP approximation of it).

---

## ⚠️ Common errors & how to fix them

| Symptom | Likely cause | Fix |
|---|---|---|
| `ValueError` / shape error from the covariance or `marginal_likelihood` | `X` passed as 1-D, shape `(n,)` | Always reshape to 2-D: `X = x[:, None]`, shape `(n, 1)`. |
| `AttributeError: 'Marginal' object has no attribute 'prior'` | Calling `.prior` on a `Marginal`/`MarginalApprox` | Use `.marginal_likelihood`. `.prior` exists only on `Latent`, `HSGP`, `TP`. |
| `LinAlgError: ... not positive definite` (during fit or `.conditional`) | Near-duplicate inputs, tiny lengthscale, insufficient jitter | Raise `jitter` (e.g. `1e-4` or `1e-3`); de-duplicate inputs; standardize `X`; ensure the lengthscale prior excludes near-zero $\ell$. |
| Many divergences; `ell` drifts toward 0; low ESS | Flat likelihood toward short lengthscales (non-identifiability) | Use an **InverseGamma** or **Gamma** lengthscale prior (Betancourt); raise `target_accept` to 0.95–0.99; standardize inputs. |
| Latent GP samples extremely slowly or gets stuck | $n$ too large for the $O(n^3)$ latent path; or centered parameterization | Switch to **HSGP** (low-D) or **MarginalApprox** (higher-D); keep `reparameterize=True`. |
| `noise=` raises "unexpected keyword" / deprecation warning | Legacy v3/v4 spelling | Use **`sigma=`** in `marginal_likelihood` (scalar, or a Covariance for correlated noise). |
| `pm.gp.MarginalSparse` not found | v3 class name | It was renamed; use **`pm.gp.MarginalApprox`** in v5. |
| HSGP fit looks truncated / biased near the edges | Data near or outside the boundary $[-L, L]$ | Increase `c` (≥ 1.2, often 2–4); make sure *prediction* points are also inside the boundary. |
| HSGP too smooth, misses real wiggles | Too few basis functions | Increase `m`; use `approx_hsgp_hyperparams(x_range, lengthscale_range, cov_func)` to size `m`, `c`. |
| Predicted sub-component of an additive GP is nonsense | Forgot the `given=` dict in the sub-component's `.conditional` | Pass `given={"X": X, "y": y, "sigma": sigma, "gp": gp_full}`. |
| Prediction interval for new $y$ looks too tight | Used `.conditional` default (`pred_noise=False`, returns latent $f$) | Pass `pred_noise=True` to get the predictive distribution of *observations*. |
| Amplitude $\eta$ and noise $\sigma$ both huge or both tiny, unstable | $\eta$/$\sigma$ trade-off (both explain scatter) | Give $\sigma$ a prior favoring some noise (Exponential/HalfNormal); standardize $y$; tighten priors. |

---

## 🧪 Exercises

1. **(Conceptual — the knobs.)** Without running anything, predict what happens to GP *prior* draws as you (a) halve the lengthscale $\ell$, (b) triple the amplitude $\eta$, (c) switch `ExpQuad` → `Matern32`. Then run the prior-draw code from §1 and confirm each prediction. Write one sentence on which knob changes wiggliness, which changes vertical scale, and which changes smoothness/roughness. *Hint: $\eta$ is the marginal SD; $\ell$ is the decorrelation distance; the Matérn $\nu$ sets differentiability.*

2. **(Coding — recover the truth.)** Re-run the §5 synthetic experiment, but generate the data with a **Matérn-5/2** kernel ($\eta=2$, $\ell=0.7$, $\sigma=0.5$) instead of ExpQuad. Fit it twice: once with the *correct* `Matern52` kernel and once with a *misspecified* `ExpQuad` kernel. Compare the recovered $\ell$, the posterior predictive bands, and `az.loo` (Chapter 11) between the two. Does the wrong kernel hurt? Where? *Hint: misspecification usually shows up as bias in $\ell$ and slightly miscalibrated bands, not catastrophic failure — GPs are forgiving.*

3. **(Coding — the lengthscale prior matters.)** Take the §5 data and fit it three times with three lengthscale priors: (a) `pm.Uniform("ell", 0.001, 20)`, (b) `pm.HalfNormal("ell", 5)`, (c) `pm.InverseGamma("ell", alpha=4, beta=4)`. Record the number of divergences, the ESS for `ell`, and whether `ell` drifts toward zero, for each. Explain the pattern using Betancourt's argument from §6. *Hint: you should see the Uniform and HalfNormal fits suffer; the InverseGamma should be clean. This is the single most instructive GP experiment you can run.*

4. **(Coding — HSGP speedup.)** Generate a synthetic 1-D GP dataset with $n = 2000$. Fit it with exact `pm.gp.Marginal` and with `pm.gp.HSGP` (use `approx_hsgp_hyperparams` to size `m`, `c`). Time both. Compare the recovered hyperparameters and the predictive curves. Then push $n$ to $20{,}000$ — does the exact GP still finish? Report the wall-clock ratio. *Hint: keep the input range fixed and `c ≥ 2`; check that your prediction grid is inside the boundary.*

5. **(Coding — latent Poisson GP.)** Load the coal-mining-disasters data (`pm.get_data("coal.csv")`) and model the *rate* of disasters over time as a **latent log-Gaussian-process**: $\lambda(t) = \exp(f(t))$, $f \sim \mathcal{GP}$, counts $\sim$ Poisson. Use `pm.gp.Latent` (or `pm.gp.HSGP` for speed). Plot the posterior intensity with credible bands. Where does the disaster rate drop, and how certain is the GP about *when*? *Hint: bin the event times into counts per interval; use an `InverseGamma` lengthscale; this is a smooth-changepoint view that we revisit in Chapters 13 and 15.*

6. **(Conceptual — choose the tool.)** For each scenario, say whether you'd reach for a GP, a spline, BART, or a parametric GLM, and why: (a) a smooth dose–response curve in 1-D with tight uncertainty needs; (b) a churn model with 60 features and many interactions; (c) interpolating a known-period daily electricity-demand signal; (d) a strictly linear relationship you're confident about. *Hint: use the rules of thumb from §10 — smoothness, dimensionality, and how much you trust the functional form.*

---

## 📚 Resources & further reading

**The textbook.**
- Rasmussen & Williams, *Gaussian Processes for Machine Learning* (MIT Press, 2006) — **free full PDF at** https://gaussianprocess.org/gpml/chapters/RW.pdf. Read **Ch. 2** (regression, the two predictive equations), **Ch. 4** (covariance functions — the kernel catalog), and **Ch. 6.3** (the GP–spline correspondence). This *is* the GP book.

**The robust-GP argument (read this before you fit a real GP).**
- Betancourt, **"Robust Gaussian Process Modeling"** — https://betanalpha.github.io/assets/case_studies/gaussian_processes.html. The canonical treatment of lengthscale non-identifiability and the InverseGamma fix — the *why* behind §6.
- Stan User's Guide, **"Fitting Gaussian Processes"** — https://mc-stan.org/docs/stan-users-guide/gaussian-processes.html. The Stan-side view (`gp_exp_quad_cov`) and the same lengthscale-prior advice; good cross-reference for the Stan path.

**PyMC documentation (authoritative signatures — verify against your installed version).**
- GP core notebook (which class has `.prior` vs `.marginal_likelihood`, the additive `given=` pattern) — https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/Gaussian_Processes.html
- `pm.gp.Marginal` API — https://www.pymc.io/projects/docs/en/stable/api/gp/generated/pymc.gp.Marginal.html
- `pm.gp.Latent` API — https://www.pymc.io/projects/docs/en/stable/api/gp/generated/pymc.gp.Latent.html
- `pm.gp.HSGP` API — https://www.pymc.io/projects/docs/en/stable/api/gp/generated/pymc.gp.HSGP.html
- `pm.gp.MarginalApprox` API — https://www.pymc.io/projects/docs/en/stable/api/gp/generated/pymc.gp.MarginalApprox.html

**Worked PyMC examples (run these end to end).**
- **Mean & Covariance Functions** (visual catalog of every kernel and composition) — https://www.pymc.io/projects/examples/en/latest/gaussian_processes/GP-MeansAndCovs.html
- **GP-Marginal** (clean regression walkthrough, `.conditional` with `pred_noise`) — https://www.pymc.io/projects/examples/en/latest/gaussian_processes/GP-Marginal.html (the `/en/latest/` build tracks current v5 APIs; if the page 404s, the older pinned `/en/2022.01.0/` build still has the walkthrough, but cross-check its code against v5)
- **GP-Latent** (Bernoulli-logit and Student-t latent GPs; the InverseGamma($\mu,\sigma$) + Exponential($\eta$) pattern) — https://www.pymc.io/projects/examples/en/latest/gaussian_processes/GP-Latent.html
- **HSGP Basic / First Steps** (rules of thumb for `m`, `c`; `approx_hsgp_hyperparams`; Gram-matrix validation) — https://www.pymc.io/projects/examples/en/latest/gaussian_processes/HSGP-Basic.html
- **HSGP Advanced Usage** (multi-dim `m=[25,25]`, boundary intuition) — https://www.pymc.io/projects/examples/en/latest/gaussian_processes/HSGP-Advanced.html
- **Mauna Loa CO₂, parts 1 & 2** (additive kernel design, component prediction, Gamma lengthscale priors) — https://www.pymc.io/projects/examples/en/latest/gaussian_processes/GP-MaunaLoa.html and https://www.pymc.io/projects/examples/en/latest/gaussian_processes/GP-MaunaLoa2.html
- **Sparse Approximations** (inducing points, choosing `Xu`, VFE vs FITC) — https://www.pymc.io/projects/examples/en/latest/gaussian_processes/GP-SparseApprox.html

**The HSGP method papers.**
- Riutort-Mayol, Bürkner, Andersen, Solin & Vehtari (2022), **"Practical Hilbert space approximate Bayesian Gaussian processes for probabilistic programming"** — https://arxiv.org/abs/2004.11408. The source of the `m`/`c`/`L` design rules.
- Solin & Särkkä (2019), **"Hilbert space methods for reduced-rank Gaussian process regression"**, *Statistics and Computing* — the original reduced-rank derivation behind `pm.gp.HSGP`.

**Connections.**
- Neal, *Bayesian Learning for Neural Networks* (1996), Ch. 2 — the infinite-width NN → GP limit.
- Lee et al., "Deep Neural Networks as Gaussian Processes" (ICLR 2018); Jacot et al., "Neural Tangent Kernel" (NeurIPS 2018) — the deep extension.
- Kimeldorf & Wahba (1970) — the GP ↔ spline correspondence (see also GPML Ch. 6.3).

**Practitioner blog.**
- Juan Orduz, **"Gaussian Processes for Time Series with PyMC"** — https://juanitorduz.github.io/gp_ts_pymc3/ — additive kernels for forecasting (update the APIs to v5 when you adapt the code).

---

## ➡️ What's next

You now have the most powerful *smooth* nonparametric model in the Bayesian toolkit — a prior over functions with calibrated uncertainty, the kernels to shape it, the priors to tame it, and the HSGP/sparse tricks to scale it. But GPs are not the only way to be flexible, and they have real blind spots: they assume one global smoothness, they struggle in high dimensions, and they do not handle sharp jumps or strong interactions gracefully. In **Chapter 13 — Mixture Models and Latent Variables** we pivot from "one smooth function" to "the data come from several hidden sub-populations" — finite mixtures, the label-switching identifiability trap and its ordered-constraint cure, marginalizing discrete latent variables (why PyMC sums them out and how Stan does it with `log_sum_exp`), and zero-inflated and hurdle models. Then **Chapter 14 — Nonparametric and Flexible Models** brings the threads of this chapter together with splines, BART, and Dirichlet-process mixtures, and makes the "GP vs spline vs BART" decision we previewed in §10 fully concrete. The GP you learned here is the smooth, calibrated benchmark every one of those models is measured against.
