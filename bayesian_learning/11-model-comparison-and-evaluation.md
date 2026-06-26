# Chapter 11 — Model Comparison & Evaluation

### How to choose between models without fooling yourself — the predictive view, LOO-CV, and the honest reading of `az.compare`

> *A note from me to you.*
>
> *You now know how to build a zoo of models — GLMs (Ch 09), hierarchical models (Ch 10) — and how to diagnose whether the sampler did its job (Ch 07). But every real analysis ends with a harder question that no $\hat R$ can answer: **which model should I actually use?** A pooled model or a hierarchical one? Poisson or negative binomial? Three predictors or five? A spline with six knots or twelve? This is the chapter where you learn to answer that question like a professional instead of a sales rep for your own favorite model.*
>
> *Here is the trap I want to inoculate you against, because almost everyone falls into it once. You fit two models, the bigger one fits the data you have **better** — lower error, higher likelihood, prettier residual plots — and you conclude the bigger model is better. It is not. It fit your training data better because **you gave it more rope to memorize the noise**, and noise does not repeat. The whole discipline of model comparison is the discipline of estimating how well a model will predict data **it has not seen**, using only the data you have. That is a genuinely deep idea — it took the field decades and a beautiful chain of results (AIC, DIC, WAIC, and finally LOO-CV via Pareto-smoothed importance sampling) to do it well — and the modern Bayesian answer is shockingly practical: one function call, one table, one plot, and a diagnostic that tells you when to stop trusting it.*
>
> *But I am also going to teach you the honesty that the table demands. The single most common sin in this material is **over-reading a tiny difference** — declaring a winner when the gap is well inside the noise. By the end of this chapter you will read an `az.compare` table the way a good doctor reads a lab panel: never one number in isolation, always the estimate next to its uncertainty, always with a diagnostic that says "this measurement is reliable" or "this one isn't." And I will show you the grown-up move that beats picking a single model at all: **stacking** them. Finally — because you will be asked — we will look squarely at **Bayes factors**, why they are so prior-sensitive that they will betray you, and the narrow circumstances where they are nonetheless the right tool. Let's learn to choose.*

---

## What you'll be able to do after this chapter

- **State the predictive philosophy of model evaluation**: we compare models by their expected **out-of-sample** predictive accuracy (ELPD), not their in-sample fit, and explain precisely why in-sample fit rewards overfitting.
- **Define ELPD** as the integral of the log predictive density against the true data-generating process, and explain why the **log score** is the principled target (a strictly proper scoring rule).
- **Explain LOO-CV and PSIS**: why exact leave-one-out requires $n$ refits, how importance sampling reuses the *single* full-data posterior to approximate all $n$ of them, and what Pareto-smoothing does to tame the heavy-tailed weights.
- **Read the Pareto $\hat k$ diagnostic** — the thresholds ($<0.5$ good, $<0.7$ ok, $>0.7$ unreliable, $\ge 1$ bad), the modern sample-size-adaptive `good_k`, and what $\hat k$ tells you about *influential* points — and know your three options when it's high (moment matching, exact refit / K-fold, or a better model).
- **Compute LOO and WAIC** in PyMC v5 + ArviZ — including the step everyone forgets, computing the pointwise `log_likelihood` — and explain why **LOO is preferred over WAIC**.
- **Read an `az.compare` table column by column** — `elpd_loo`, `p_loo` (effective parameters, itself a diagnostic), `elpd_diff`, `dse` (the SE *of the difference*), and `weight` — and apply the "$|{\rm elpd\_diff}| < 4$ is noise" rule honestly.
- **Average models with stacking and pseudo-BMA+** (Yao et al. 2018) instead of betting everything on one, and say when each is appropriate (M-open vs M-closed).
- **Check calibration with LOO-PIT** and use posterior predictive checks as the qualitative complement to the numeric ELPD.
- **Explain why Bayes factors are fragile** — extreme prior sensitivity, the Jeffreys–Lindley paradox, undefined under improper priors — and when bridge sampling is nonetheless the right tool.

I assume you are fluent through **Chapter 10 (Hierarchical & Multilevel Models)**: you can write a `pm.Model`, set standardized weakly-informative priors (Ch 03), sample to convergence and read the diagnostics from **Chapter 07** ($\hat R \le 1.01$, ESS $\gtrsim 400$, zero divergences), and run a posterior predictive check from **Chapter 08**. Model comparison is the **"model expansion / comparison" step of the Bayesian workflow** (Ch 08); this chapter is where we make it rigorous.

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

## 1. The whole idea: we predict the future, not the past

Let me put the entire chapter on one slide before we earn it.

You have data $y = (y_1, \dots, y_n)$ and two or more candidate models. You want the one that will make the best predictions about **new** data drawn from the same process. You do not have new data. So you have to **estimate** out-of-sample predictive performance using only $y$. Every method in this chapter — LOO, WAIC, AIC, DIC — is an estimator of one quantity, the **expected log pointwise predictive density (ELPD)**, and they differ only in how cleverly and how cheaply they estimate it.

That's it. The rest is detail, but the detail is where the money is.

> 💡 **Intuition:** A model is a small world (McElreath's phrase from Ch 01). Comparison asks: *of these small worlds, which one will best anticipate the large world's next observation?* Not "which one drew the prettiest picture of the observations I already have" — any sufficiently flexible model can draw a perfect picture of the past by memorizing it. We want the world that **generalizes**.

### Why in-sample fit is a liar

Consider the simplest possible demonstration. I'll generate data from a gentle quadratic and fit polynomials of increasing degree. Watch what happens to in-sample fit (residual sum of squares, or equivalently the maximized log-likelihood) as I add terms.

```python
# A controlled overfitting demo: data from a quadratic, fit polynomials of degree 1..9
n = 20
x = np.linspace(-1, 1, n)
true_mu = 0.5 + 1.2 * x - 0.8 * x**2
y_demo = true_mu + rng.normal(0, 0.25, size=n)

insample_r2 = []
for degree in range(1, 10):
    X = np.vander(x, degree + 1)          # design matrix with columns 1, x, x^2, ...
    beta, *_ = np.linalg.lstsq(X, y_demo, rcond=None)
    yhat = X @ beta
    ss_res = np.sum((y_demo - yhat) ** 2)
    ss_tot = np.sum((y_demo - y_demo.mean()) ** 2)
    insample_r2.append(1 - ss_res / ss_tot)

for d, r2 in zip(range(1, 10), insample_r2):
    print(f"degree {d}:  in-sample R^2 = {r2:.3f}")
```

The in-sample $R^2$ climbs **monotonically** toward 1.0 as the degree grows. By degree 9 (ten parameters for twenty points) the curve threads nearly every observation and $R^2 \approx 1$. If you chose your model by in-sample fit you would always choose the most flexible model on offer — and you would always be wrong, because that wiggly degree-9 curve has fit the *noise*, the part that will not repeat in new data. Run the same models forward to fresh $y$ from the same process and the high-degree fits are a catastrophe.

> ⚠️ **Pitfall:** In-sample log-likelihood, $R^2$, training MSE, and "the residuals look great" are all **monotone or near-monotone in model flexibility**. They cannot, even in principle, tell you to stop adding complexity. You need an estimate of *out-of-sample* performance, which is what the rest of this chapter builds.

> 🧮 **The math (the bias–variance decomposition, Bayesian flavor):** Adding parameters lowers the model's *bias* (it can represent more functions) but raises its *variance* (the fit responds more to the particular noise in this dataset). In-sample error keeps falling because it only sees the bias term shrink; it is blind to the variance you're buying. Out-of-sample error is the sum of both and is U-shaped in flexibility — it falls, bottoms out at the right complexity, then rises. We are trying to find the bottom of that U **without seeing the test set.**

### The predictive distribution is the thing we score

In a Bayesian model we don't predict with a single fitted parameter; we predict with the **posterior predictive distribution**, which averages the likelihood over the whole posterior:

$$
p(\tilde y \mid y) \;=\; \int p(\tilde y \mid \theta)\, p(\theta \mid y)\, d\theta .
$$

This is the object whose quality we want to measure. It already bakes in parameter uncertainty — a model that is unsure about $\theta$ produces a wider, more honest predictive distribution, and a good scoring rule will reward that honesty rather than punish it. That's important: we are not scoring a point prediction, we are scoring a **predictive distribution**, and the right way to score a predictive distribution is the log score.

---

## 2. ELPD: the quantity everything estimates

Here is the target, stated precisely. Imagine you could draw a brand-new dataset $\tilde y$ from the **true** data-generating process $p_t$ (which you never get to see). The expected log pointwise predictive density is

$$
\mathrm{elpd} \;=\; \sum_{i=1}^{n} \int p_t(\tilde y_i)\, \log p(\tilde y_i \mid y)\, d\tilde y_i .
$$

In words: for each of the $n$ future observations, take the log of the predictive density your model assigns to it, and average over where the *true* process actually puts that future observation. Sum over the $n$ points. **Higher ELPD is better** — your predictive distribution put high density exactly where the future data landed.

In ASCII, point by point:

```
elpd = sum_i  E_{ytilde_i ~ p_true} [ log p(ytilde_i | y) ]
       ^^^^^^                          ^^^^^^^^^^^^^^^^^^^^^
       over all n future points        log predictive density at the true future value
```

> 💡 **Intuition:** ELPD rewards a model for putting probability mass where reality will actually land — and, crucially, for being **honestly uncertain** elsewhere. Assign razor-sharp density to the wrong place and the log score punishes you brutally (the log of a tiny number is a large negative). Spread your bets sensibly and you're rewarded. This is exactly the behavior we want when choosing a model to deploy.

### Why the *log* score, specifically?

You could imagine scoring predictions a hundred ways — squared error, absolute error, classification accuracy. Why does the whole Bayesian model-comparison apparatus insist on the log of the predictive density? Because the log score is a **strictly proper scoring rule**, and that property is exactly what you want.

> 🧮 **The math (proper scoring rules):** A scoring rule $S(q, y)$ scores a predictive distribution $q$ against an outcome $y$. It is **proper** if the expected score is optimized (here, maximized) by reporting the *true* distribution — that is, if $p_t$ is the truth, then $\mathbb{E}_{y \sim p_t}[S(p_t, y)] \ge \mathbb{E}_{y \sim p_t}[S(q, y)]$ for every $q$. It is **strictly** proper if equality holds only when $q = p_t$. The log score $S(q, y) = \log q(y)$ is strictly proper. The practical payoff: a strictly proper rule cannot be gamed. There is no way to get a better expected score than by reporting your honest predictive distribution — you can't win by being overconfident, and you can't win by hedging vaguely. A model selection criterion built on a non-proper rule (accuracy is a notorious example) can be *gamed* by a miscalibrated model, which is exactly the failure mode we're trying to avoid.

> 📜 **Citation/Origin:** The decision-theoretic case for the log score and proper scoring rules is Gneiting & Raftery (2007), "Strictly Proper Scoring Rules, Prediction, and Estimation." The use of ELPD as *the* target for Bayesian model evaluation, with LOO and WAIC as its estimators, is laid out in **Vehtari, Gelman & Gabry (2017)**, "Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC" — the paper this chapter's formulas come from. Read its §2–3.

### The catch, and the bridge to LOO

We cannot compute ELPD because we don't have $p_t$ — we don't get future data. So we estimate it from the data we *do* have. The naive estimate is the **in-sample** version: just score the model on the same $y$ it was fit to,

$$
\mathrm{lppd} \;=\; \sum_{i=1}^n \log p(y_i \mid y) \;=\; \sum_{i=1}^n \log \int p(y_i \mid \theta)\, p(\theta \mid y)\, d\theta ,
$$

the **log pointwise predictive density**. But this is the liar from §1 — it uses each $y_i$ twice (once to fit, once to score), so it is **optimistically biased**: it overstates how well the model will do on genuinely new data, and the overstatement grows with model flexibility. Every method in this chapter is, at heart, **lppd minus a correction for that optimism**:

- **AIC** subtracts the number of parameters $p$ (only valid for flat priors and MLE).
- **DIC** subtracts an effective-parameter count from a plug-in posterior mean (not fully Bayesian; deprecated).
- **WAIC** subtracts a fully-Bayesian, pointwise effective-parameter penalty $p_{\text{waic}}$.
- **LOO-CV** sidesteps the correction entirely by **actually leaving each point out** — and is the method we trust most.

Let's build LOO properly, because it's the workhorse.

---

## 3. LOO-CV: leave one out, score it, repeat — but cheaply

**Cross-validation** is the most honest answer to "how will this predict new data?": hold some data out, fit on the rest, score on the held-out part, repeat. **Leave-one-out** cross-validation takes this to its limit — hold out *exactly one* observation at a time. The LOO estimate of ELPD is

$$
\mathrm{elpd}_{\mathrm{loo}} \;=\; \sum_{i=1}^n \log p(y_i \mid y_{-i}),
\qquad
p(y_i \mid y_{-i}) \;=\; \int p(y_i \mid \theta)\, p(\theta \mid y_{-i})\, d\theta ,
$$

where $y_{-i}$ means "all the data except point $i$." Read it carefully: for each point, you fit the model on the *other* $n-1$ points, then ask how much density the resulting posterior predictive assigns to the point you held out. Because point $i$ was never used to fit the posterior $p(\theta \mid y_{-i})$, this is an **honest out-of-sample** score for that point. Sum over all points and you have an almost-unbiased estimate of ELPD. No optimism, no hand-tuned penalty.

> 💡 **Intuition:** LOO is "the model grades itself on questions it didn't study for," $n$ times, once per question. It is the gold standard precisely because it never lets a point grade its own answer.

### The problem: $n$ refits is insane

The formula asks you to compute $n$ different posteriors, $p(\theta \mid y_{-i})$ for every $i$. For a model that takes a minute to sample and a dataset of 10,000 points, that's a week of compute for *one* model comparison. Completely infeasible. This is why people reached for cheaper analytic approximations (AIC, DIC, WAIC) for decades.

The breakthrough is to realize you already have something very close to all $n$ leave-one-out posteriors: the **full-data posterior** $p(\theta \mid y)$ you already sampled. Leaving one point out barely perturbs the posterior (it's one observation out of $n$). So instead of re-sampling, we **re-weight** the draws we already have. That's importance sampling.

### Importance sampling: reuse the full posterior

We want expectations under the leave-one-out posterior $p(\theta \mid y_{-i})$, but we only have draws $\theta^{(1)}, \dots, \theta^{(S)}$ from the full posterior $p(\theta \mid y)$. Importance sampling corrects for the mismatch by weighting each draw by the ratio of the two densities. By Bayes' rule the two posteriors differ by exactly one likelihood factor — the contribution of point $i$ — so the importance ratio for draw $s$ at point $i$ is

$$
r_i^{(s)} \;\propto\; \frac{p(\theta^{(s)} \mid y_{-i})}{p(\theta^{(s)} \mid y)} \;\propto\; \frac{1}{p(y_i \mid \theta^{(s)})}.
$$

That is a wonderfully cheap quantity: it's just **one over the likelihood of point $i$ at draw $s$** — and you computed those likelihoods anyway. The leave-one-out predictive density becomes a weighted average over your existing draws:

$$
p(y_i \mid y_{-i}) \;\approx\; \frac{\sum_s r_i^{(s)}\, p(y_i \mid \theta^{(s)})}{\sum_s r_i^{(s)}}
\;=\; \frac{\sum_s 1}{\sum_s 1/p(y_i\mid\theta^{(s)})}
\;=\; \left( \frac{1}{S}\sum_{s} \frac{1}{p(y_i \mid \theta^{(s)})} \right)^{\!-1}.
$$

So in principle: take the full posterior, form the per-point likelihoods, take reciprocals, and you get all $n$ leave-one-out scores from **one fit**. Magic — except for one problem.

> ⚠️ **Pitfall:** Those raw importance ratios $1/p(y_i \mid \theta^{(s)})$ have **heavy tails**. The worst offenders are draws where the model assigned tiny likelihood to point $i$ — exactly the draws where $1/p$ explodes. A handful of enormous weights dominate the sum, the estimate's variance blows up, and the whole thing becomes unreliable and unstable. Plain importance-sampling LOO (sometimes "IS-LOO") is *not* safe to use raw. This is the gap PSIS fills.

### PSIS: Pareto-smoothed importance sampling

Here is the elegant fix, and it's the reason modern LOO is trustworthy. The largest importance weights are the problem, and a beautiful piece of extreme-value theory says the tail of a well-behaved importance-weight distribution looks like a **generalized Pareto distribution (GPD)**. PSIS does the following, per observation $i$:

1. Take the importance ratios $r_i^{(1)}, \dots, r_i^{(S)}$.
2. Fit a generalized Pareto distribution to the **largest** ratios (the upper tail — by default the top 20% or so).
3. **Replace** those largest ratios with the expected order statistics of the fitted GPD — i.e., smooth the wild tail into well-behaved values, and cap them.
4. Use these smoothed, stabilized weights to form the weighted average above.

This dramatically reduces the variance of the estimate while keeping it nearly unbiased. And — this is the part I want you to fall in love with — the fitting in step 2 produces a **free diagnostic** that tells you when even PSIS couldn't save that point.

> 📜 **Citation/Origin:** PSIS is **Vehtari, Simpson, Gelman, Yao & Gabry (2024)**, "Pareto Smoothed Importance Sampling" (JMLR; preprint arXiv:1507.02646). PSIS-LOO as a model-evaluation method is the 2017 *Stat. Comput.* paper. The implementation you call through ArviZ is the Python sibling of the Stan team's `loo` R package.

---

## 4. The Pareto $\hat k$ diagnostic: your trust meter

When PSIS fits that generalized Pareto distribution to the tail of point $i$'s importance ratios, it estimates the GPD's **shape parameter**, which we call $\hat k_i$. This single number per observation is the most important diagnostic in this whole chapter, because it measures **how heavy the tail of the importance weights was** — equivalently, how badly the full-data posterior had to be re-weighted to approximate the leave-one-out posterior for that point.

> 🧮 **The math (why $\hat k$ is a tail-heaviness meter):** The shape parameter $k$ of a generalized Pareto distribution controls how many moments are finite: the distribution has finite moments up to order $\lfloor 1/k \rfloor$. So $k < 0.5$ means finite variance (the importance-sampling estimate has a central limit theorem and converges nicely), $0.5 \le k < 1$ means finite mean but infinite variance (convergence is slow and the estimate is noisy), and $k \ge 1$ means the importance-sampling estimand **doesn't even have a finite mean** — PSIS is fundamentally ill-defined for that point. That is the precise reason the thresholds fall where they do.

> 🩺 **Diagnostic: reading $\hat k$.** ArviZ gives you a $\hat k_i$ for every observation when you pass `pointwise=True`. The classic fixed thresholds — the ones to memorize, and the ones we use consistently across this course (Ch 07, Appendix A):
>
> | $\hat k$ range | Verdict | What it means |
> |---|---|---|
> | $\hat k < 0.5$ | **Good** | PSIS estimate is reliable; fast convergence. |
> | $0.5 \le \hat k < 0.7$ | **OK** | Still usable; estimate slightly noisier. |
> | $0.7 \le \hat k < 1$ | **Unreliable** | PSIS has large bias for this point; do not trust its LOO contribution. |
> | $\hat k \ge 1$ | **Bad** | The LOO estimand has non-finite mean; PSIS is meaningless here. |
>
> The single rule to carry: **$\hat k < 0.7$ for trustworthy PSIS-LOO.** Above that, the estimate for that point — and therefore the whole `elpd_loo` — is suspect.

### The modern, sample-size-adaptive threshold

Recent ArviZ (following Vehtari et al. 2024) refines the fixed 0.7 into a threshold that *depends on how many posterior draws you have*, because more draws can rescue a moderately heavy tail. ArviZ exposes it as `good_k` on the LOO result:

$$
\hat k_{\text{threshold}} \;=\; \min\!\left(1 - \frac{1}{\log_{10} S},\; 0.7\right),
$$

where $S$ is the number of posterior draws. With the usual few thousand draws the raw value $1 - 1/\log_{10} S$ actually *exceeds* 0.7 (e.g. $S=4000 \to 0.722$, $S=8000 \to 0.744$), so the `min(·, 0.7)` cap binds and `good_k` is **exactly 0.7** — which is precisely why remembering the fixed 0.7 costs you nothing. Only with *fewer* than ~2200 draws does the adaptive threshold drop below 0.7. The nuance worth knowing: a point whose $\hat k$ sits between this adaptive threshold and 0.7 may become reliable if you **draw more samples** (roughly past $S > 2200$), whereas a $\hat k$ above 0.7 is a property of the *model and that data point*, not of how long you sampled — more draws won't help.

> 💡 **Intuition:** A high $\hat k_i$ flags an **influential observation** — a point so surprising under the model that removing it would noticeably move the posterior. These are your outliers, your high-leverage points, the observations the model is straining to accommodate. So $\hat k$ does double duty: it's a *reliability* meter for LOO **and** an *influence* meter for your data. A cluster of high-$\hat k$ points is the model whispering "these observations don't fit my story."

### What to do when $\hat k$ is high

You will eventually run `az.loo` and get the dreaded warning *"Estimated shape parameter of Pareto distribution is greater than 0.7."* Here is your decision tree, in order of escalating effort:

1. **A few points just over 0.7?** First, *look at them.* Identify which observations they are (the worked example shows how). Are they genuine outliers, data errors, or a sign the model is misspecified? Often a posterior predictive check (§9) will independently flag the same points. Sometimes the right fix is **a better model** (e.g., switch a Normal likelihood to Student-$t$, as in Ch 09 — robust models have lower $\hat k$ because no single point is shocking).

2. **Want to rescue PSIS for those points?** Use **moment matching**, which cheaply nudges the posterior toward each bad point's leave-one-out posterior (matching the first couple of moments) without a full refit, salvaging many borderline points. Note the API reality: ArviZ's `az.loo` does **not** expose this — there is no moment-matching argument on it, and in particular passing `reff=` does *not* trigger it (`reff` is only the relative MCMC efficiency; see §6). Moment matching lives in the `loo` R package's `loo_moment_match()` and is being worked into the PyMC/ArviZ ecosystem; for Python today, the practical escalation is the exact refit / K-fold of step 3.

3. **Want the exact answer for those points?** **Refit them.** The "reloo" idea — an exact refit of *only* the high-$\hat k$ points (usually a handful), substituting each one's exact LOO contribution — has no single canonical function in PyMC/ArviZ today; you implement it as a small manual refit loop over the flagged indices (the same recipe as K-fold in §10, just restricted to the bad points). When *many* points are bad, escalate to full **K-fold cross-validation**: partition the data into K folds, refit K times, accumulate the held-out pointwise log-likelihoods. ArviZ won't auto-refit for you — you loop the folds yourself (§10 sketches this). K-fold is the universal fallback that always works; it's just more expensive.

> 🔧 **In practice:** 95% of the time, a clean model on reasonable data gives you all-green $\hat k$ and you never think about this again. The 5% where $\hat k$ lights up is *information*, not just an annoyance — it's pointing at the observations and the modeling decisions that matter most. Treat a wall of high $\hat k$ as a prompt to improve the model, not merely to switch to K-fold.

---

## 5. WAIC, and why we still prefer LOO

Before PSIS made LOO cheap, the fully-Bayesian tool of choice was the **Widely Applicable Information Criterion (WAIC)**, due to Watanabe. It's worth understanding because you'll see it in papers, ArviZ computes it (`az.waic`), and it makes the "lppd minus an optimism penalty" structure from §2 completely explicit:

$$
\widehat{\mathrm{elpd}}_{\mathrm{waic}}
\;=\;
\underbrace{\sum_{i=1}^n \log\!\Big(\tfrac{1}{S}\sum_{s=1}^S p(y_i\mid\theta^{(s)})\Big)}_{\textstyle \mathrm{lppd}\ (\text{in-sample fit})}
\;-\;
\underbrace{\sum_{i=1}^n \mathrm{Var}_s\big[\log p(y_i\mid\theta^{(s)})\big]}_{\textstyle p_{\mathrm{waic}}\ (\text{effective \# parameters})}.
$$

The first term is the in-sample lppd — the optimistic estimate. The second term, $p_{\text{waic}}$, is the **penalty**: for each point, the *variance across posterior draws* of its log-likelihood. The intuition is gorgeous: if the posterior is very *uncertain* about how well it predicts point $i$ (high variance in $\log p(y_i \mid \theta^{(s)})$ across draws), that point is effectively being fit by a flexible, data-hungry part of the model, so it contributes more to the overfitting penalty. Summed up, that variance behaves like a count of effective parameters — and Watanabe proved this estimator is asymptotically equal to LOO.

> 🧮 **The math (asymptotic equivalence):** As $n \to \infty$, $\widehat{\mathrm{elpd}}_{\mathrm{waic}}$ and $\widehat{\mathrm{elpd}}_{\mathrm{loo}}$ converge to the same value, and both are consistent estimators of the true ELPD. So in large, well-behaved problems they agree. The differences that matter are in **finite samples** and under **weak priors / influential points** — exactly the regimes where real analyses live.

**So why does this course prefer LOO?** Three concrete reasons:

1. **LOO carries its own failure detector.** PSIS-LOO hands you $\hat k$ per point — a precise, per-observation statement of *when the estimate is unreliable*. WAIC's only built-in warning is "the pointwise log-likelihood variance exceeded ~0.4 somewhere," which is coarse and tends to fire after the damage is done. With LOO you *know* which points broke it; with WAIC you mostly find out that something did.

2. **LOO is more robust in finite samples and with weak priors.** WAIC's $p_{\text{waic}}$ penalty is itself a variance estimate, and variance estimates are noisy; with small $n$ or flexible models it can become unstable and WAIC turns over-optimistic. PSIS-LOO degrades more gracefully.

3. **They're the same when it's easy and different when it's hard — and you want the trustworthy one when it's hard.** If WAIC and LOO agree, great, no decision to make. If they *disagree*, that disagreement is itself a signal of an unstable regime, and LOO (with $\hat k$ telling you why) is the one to trust.

> 🔧 **In practice:** Default to **`az.loo`**. Compute `az.waic` if you want the cross-check or a reviewer asks for it. If they disagree noticeably, investigate the high-$\hat k$ points and the high-$p_{\text{waic}}$ points (often the same observations), and trust LOO. McElreath's *Statistical Rethinking* Ch 7 ("Ulysses' Compass") tells this whole story — regularization, information criteria, WAIC and PSIS — and is the best narrative companion to this section.

### `p_loo`: the effective-parameters number is itself a diagnostic

LOO has its own effective-parameter count, defined as the gap between the in-sample lppd and the cross-validated estimate:

$$
p_{\mathrm{loo}} \;=\; \mathrm{lppd} - \widehat{\mathrm{elpd}}_{\mathrm{loo}}.
$$

This is the "optimism" we corrected for — how much the in-sample fit overstated things. It usually lands near the model's actual number of parameters (for a well-specified model with reasonable priors), and reading it is a quietly powerful diagnostic:

> 🩺 **Diagnostic: $p_{\text{loo}}$ as a misspecification meter.** Let $p$ be the model's actual parameter count and $n$ the number of observations.
>
> - **Healthy:** $p_{\text{loo}} < p$ and $p_{\text{loo}} \ll n$. Regularizing priors are doing their job (the effective count is *below* the nominal count — shrinkage in action; see Ch 10).
> - **Many $\hat k > 0.7$ AND $p_{\text{loo}} \ll p$:** strong sign of **model misspecification or outliers**. The model is straining against a few points it can't represent; a posterior predictive check will usually flag the same trouble. *Fix the model*, don't just paper over LOO.
> - **$p_{\text{loo}} \approx p$ (or larger) AND high $\hat k$:** a genuinely **flexible-but-reasonable model** (e.g., a hierarchical model with many group effects, or a GP). LOO is just hard here; reach for **K-fold**.
>
> The `loo` R package glossary (linked in Resources) is the definitive reference for these readings, and the logic transfers verbatim to ArviZ.

---

## 6. Computing it in PyMC + ArviZ — including the step everyone forgets

Now the practical bit. There is exactly one gotcha that trips up every newcomer, so let me put it in lights:

> ⚠️ **Pitfall — the #1 LOO error:** PyMC v5 does **not** store the pointwise log-likelihood by default. If you call `az.loo(idata)` on a fresh `pm.sample` result, you'll get a `TypeError`/`KeyError` complaining that **`log_likelihood` was not found in the InferenceData**. LOO and WAIC are *built from* the per-observation log-likelihood, so you must compute it first.

There are two supported ways, and you should know both:

```python
# (A) Compute log-likelihood AT SAMPLE TIME — pass it through idata_kwargs.
idata = pm.sample(
    1000, tune=1000, chains=4, target_accept=0.9,
    random_seed=RANDOM_SEED,
    idata_kwargs={"log_likelihood": True},   # <-- the line everyone forgets
)

# (B) Compute it AFTER THE FACT — useful if you already sampled, or sampled on a backend
#     that didn't store it. Must run inside the model context (or pass model=...).
with model:
    pm.compute_log_likelihood(idata)   # extends idata in place, adds a `log_likelihood` group
```

The verified PyMC v5 signature, for reference:

```python
pm.compute_log_likelihood(
    idata, *, var_names=None, extend_inferencedata=True,
    model=None, sample_dims=("chain", "draw"),
    progressbar=True, compile_kwargs=None,
)
```

Either way, your InferenceData ends up with a `log_likelihood` group, and ArviZ reads from it. Then LOO is one call:

```python
loo = az.loo(idata, pointwise=True)   # pointwise=True so we get per-observation pareto_k
print(loo)
loo.pareto_k        # xarray DataArray: one k-hat per observation
```

The verified `az.loo` signature (ArviZ 0.21+):

```python
az.loo(data, pointwise=None, var_name=None, reff=None, scale=None)
# scale ∈ {"log" (default; higher=better), "negative_log", "deviance"}
# reff = relative MCMC efficiency; auto-computed from ESS if omitted
# var_name = which observed variable, if your model has several
```

An `az.loo` result (an `ELPDData`, which is a `pandas.Series` subclass) prints something like the following (these illustrative numbers are from a *separate* run on an 8-observation setup — don't try to align them with the §7 worked-example table; your seed will differ):

```
Computed from 4000 posterior samples and 8 observations log-likelihood matrix.

         Estimate       SE
elpd_loo   -30.78     1.07
p_loo        0.95        -

Pareto k diagnostic values:
                         Count   Pct.
(-Inf, 0.70]   (good)        8  100.0%
   (0.70, 1]   (bad)         0    0.0%
   (1, Inf)   (very bad)     0    0.0%
```

> 🩺 **Diagnostic: reading the `az.loo` printout.** `elpd_loo` is your headline number (remember: on the default `log` scale, **higher is better**; it's negative *here* because each point's predictive density is below 1, so you're summing logs of sub-1 numbers — but with very sharp continuous predictions a density can exceed 1 and the elpd can be positive, so read the sign as relative-to-other-models, not as an absolute law). `SE` is its standard error across the $n$ pointwise contributions — your first sanity check on how *certain* the estimate is. `p_loo` is the effective-parameters diagnostic from §5. The **Pareto k table** at the bottom is the all-clear (or alarm): here every point is in the good bucket, so we trust this LOO completely.

> 🔧 **In practice — getting `log_likelihood` for free.** If you fit with **Bambi** (Ch 09), `model.fit()` already returns an InferenceData *with* the log-likelihood group, so `az.loo` works immediately. If you fit on the NumPyro or nutpie backends (Ch 17) via `pm.sample(nuts_sampler="numpyro")`, double-check the `log_likelihood` group exists and call `pm.compute_log_likelihood` if it doesn't.

> ⚠️ **Pitfall — `pm.Potential` likelihoods.** `compute_log_likelihood` only knows how to score **observed random variables** (things you declared with `observed=`). If you encoded your likelihood as a bare `pm.Potential`, there is no observed RV to score and the `log_likelihood` group will be missing or wrong. Fix: express the likelihood as an observed distribution — `pm.SomeDist("y", ..., observed=y_obs)` — or, for a genuinely custom density, use `pm.CustomDist` with an explicit `logp`, which *is* scoreable. This bites people constantly in mixture and censored models (Ch 13, 16).

---

## 7. Worked example: pooled vs hierarchical on Eight Schools

Time to put it together on a named dataset. We'll use **Eight Schools** — the canonical hierarchical-modeling example you met in Ch 07 (where it taught us the funnel and non-centering) and Ch 10 (partial pooling). It's perfect here because it gives us a clean, honest model comparison with a genuinely *interesting* answer: the more complex model does **not** dominate, and learning to report that honestly is the entire point.

The data: eight schools ran a coaching program; for each we have an estimated treatment effect $y_j$ and its standard error $\sigma_j$ (the schools differ in sample size, so the SEs differ). The question is how much the schools' true effects differ from one another.

> 🔧 **A note on scale.** Unlike most chapters, we deliberately do **not** standardize here (departing from the Ch 03 / bible default). The $y_j$ are already interpretable treatment-effect estimates with *known* standard errors $\sigma_j$, and the model uses those $\sigma_j$ directly as the likelihood scale — standardizing would throw that hard-won structure away. When your response carries its own known measurement error, fit on its native scale.

```python
# Eight Schools data (hardcoded — the canonical numbers)
y     = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
sigma = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])
J = len(y)
schools = [f"school_{i}" for i in range(J)]
coords = {"school": schools}
```

We'll compare two models that bracket the modeling spectrum:

- **Complete pooling** — assume all schools share *one* true effect $\mu$. Maximally simple, maximally biased if schools really differ.
- **Hierarchical (partial pooling)** — each school has its own effect $\theta_j$, but they're drawn from a common distribution $\mathrm{Normal}(\mu, \tau)$ whose spread $\tau$ is learned from the data. This is the Ch 10 workhorse; we write it **non-centered** (the standard cure for the funnel from Ch 07/10).

> 🔧 **In practice:** A *third* natural candidate is **no pooling** (each school estimated independently, $\theta_j$ with flat-ish priors and no shared $\tau$). I'll leave it as an exercise; with eight noisy schools it overfits and LOO will say so.

### The two models

```python
# Model 1: complete pooling — one shared effect mu
with pm.Model(coords=coords) as pooled_model:
    mu = pm.Normal("mu", mu=0, sigma=10)
    # each school's observed estimate is noisy around the SAME mu, with known sigma_j
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y, dims="school")
    idata_pooled = pm.sample(
        1000, tune=1000, chains=4, target_accept=0.9,
        random_seed=RANDOM_SEED, idata_kwargs={"log_likelihood": True},
    )

# Model 2: hierarchical, NON-CENTERED (the funnel cure from Ch 07/10)
with pm.Model(coords=coords) as hier_model:
    mu  = pm.Normal("mu", mu=0, sigma=10)
    tau = pm.HalfNormal("tau", sigma=10)            # between-school SD; HalfNormal per Ch 03
    z   = pm.Normal("z", mu=0, sigma=1, dims="school")   # standardized offsets
    theta = pm.Deterministic("theta", mu + tau * z, dims="school")  # non-centered
    y_obs = pm.Normal("y_obs", mu=theta, sigma=sigma, observed=y, dims="school")
    idata_hier = pm.sample(
        1000, tune=1000, chains=4, target_accept=0.95,   # 0.95: hierarchical funnels (Ch 07)
        random_seed=RANDOM_SEED, idata_kwargs={"log_likelihood": True},
    )
```

Two things to notice, both course-consistency rules. First, I passed `idata_kwargs={"log_likelihood": True}` on **both** `pm.sample` calls — without it, the LOO calls below would error. Second, I raised `target_accept` to `0.95` on the hierarchical model because hierarchical funnels are divergence-prone (Ch 07); on the pooled model `0.9` is plenty.

> 🩺 **Diagnostic — gate on convergence FIRST.** Before *any* model comparison, both models must be clean. Run the Ch 07 checks on each: `az.summary(idata_hier, var_names=["mu", "tau"])` should show $\hat R \le 1.01$ and ESS $\gtrsim 400$; check `idata_hier.sample_stats["diverging"].sum()` is 0 (or negligible). **Comparing two models when one didn't converge is comparing one good answer to garbage** — the LOO of an unconverged model is meaningless. The non-centered parameterization is what keeps `tau` near zero from collapsing the sampler.

### Compute LOO for each

```python
loo_pooled = az.loo(idata_pooled, pointwise=True)
loo_hier   = az.loo(idata_hier,   pointwise=True)

print("POOLED:\n",       loo_pooled, "\n")
print("HIERARCHICAL:\n", loo_hier)
```

Check the Pareto-$k$ tables first. With only eight observations and a hierarchical model, it would not be shocking to see one or two $\hat k$ creep toward 0.7 (eight points means each one is influential — removing 1/8 of the data moves the posterior a fair bit). If a $\hat k$ exceeds 0.7, note *which school* it is and treat that LOO contribution with suspicion; with so few points, K-fold (here, leave-one-school-out refit eight times) is cheap and is the gold-standard fallback.

### The comparison table

Now the headline call — `az.compare` takes a dict of models (you can pass the InferenceData objects directly, or the pre-computed `az.loo` results):

```python
compare_df = az.compare(
    {"hierarchical": idata_hier, "pooled": idata_pooled},
    ic="loo",            # use LOO (default & recommended); "waic" also available
    # method="stacking" is the default and only affects the `weight` column
)
print(compare_df.columns.tolist())   # ALWAYS print columns first — names vary by ArviZ version
compare_df
```

> ⚠️ **Pitfall — column names drift across ArviZ versions.** Older ArviZ printed `loo, p_loo, d_loo, weight, se, dse, warning, loo_scale`; recent versions use the IC-agnostic spellings `rank, elpd_loo, p_loo, elpd_diff, weight, se, dse, warning, scale`. **The semantics are stable** (best model = rank 0, `elpd_diff` is relative to the best, `dse` is the SE of that difference) but the exact strings are not — so `print(compare_df.columns.tolist())` before you index columns in code. I'll use the recent spellings below.

A representative result (your exact numbers will vary slightly by seed):

| | rank | elpd_loo | p_loo | elpd_diff | weight | se | dse | warning | scale |
|---|---|---|---|---|---|---|---|---|---|
| **pooled** | 0 | -30.6 | 0.9 | 0.00 | 0.78 | 1.0 | 0.00 | False | log |
| **hierarchical** | 1 | -30.9 | 2.6 | 0.34 | 0.22 | 1.1 | 0.42 | False | log |

Now read it like a doctor.

> 🩺 **Diagnostic — reading `az.compare`, column by column.**
>
> - **`rank`** — models sorted best (0) to worst by `elpd_loo`. Here *pooled* edges out *hierarchical*. Do **not** stop reading here; rank without the difference and its uncertainty is meaningless.
> - **`elpd_loo`** — the estimated out-of-sample log predictive density (higher = better, on `log` scale). $-30.6$ vs $-30.9$ — a difference of **0.3**.
> - **`p_loo`** — effective parameters (§5). Pooled $\approx 0.9$ (it has one parameter, $\mu$ — checks out). Hierarchical $\approx 2.6$ (one $\mu$, one $\tau$, and eight partially-pooled $\theta_j$ that *together* count as only ~0.6 effective extra parameters because $\tau$ shrank them hard toward $\mu$ — that's partial pooling, quantified). Both are well below $n=8$: healthy.
> - **`elpd_diff`** — each model's `elpd_loo` minus the best model's. Best is 0 by construction; hierarchical is **0.34** worse.
> - **`dse`** — the standard error of *that difference*, computed pairwise on the **pointwise** elpd contributions. This is the number that matters, and it's $\approx 0.42$ here. Crucially `dse` is **not** `se` of the two models combined — because the two models make correlated errors on the same points, the difference is estimated far more precisely than treating the `se` columns as independent would suggest.
> - **`weight`** — the stacking weight (§8), loosely "how much you'd lean on this model when averaging." Pooled gets most of it.
> - **`warning`** — `True` if any $\hat k$ was bad; check it every time.

### The honest verdict

Here is the discipline this whole chapter exists to instill. The hierarchical model is `elpd_diff = 0.34` worse, with `dse = 0.42`. Apply the rule:

> 🩺 **The decision rule (Vehtari CV-FAQ).**
> 1. If **$|{\rm elpd\_diff}| < 4$**, the difference is **small and noisy** — *do not over-interpret it, and do not select on it.* Four is a rough "below this, the log-score gap is in the weeds" line.
> 2. If $|{\rm elpd\_diff}| > 4$, then compare it to `dse`: a difference of several `dse` (loosely, ${\rm elpd\_diff}/{\rm dse} \gtrsim 2$ read as a z-score) is **meaningful**.

Our `elpd_diff = 0.34` is far below 4, and it's *smaller than its own `dse`* (0.34 < 0.42). The correct, professional conclusion is:

> **"The two models are predictively indistinguishable on this data."** Neither pooled nor hierarchical wins. We are not entitled to say "pooled is better" — the gap is pure noise.

And that is exactly right and exactly *interesting*: with eight schools whose effects are not wildly different, partial pooling shrinks the $\theta_j$ so hard toward the common mean that the hierarchical model behaves almost like the pooled one — so they predict almost identically. The honest report is "either is fine; I'll keep the hierarchical model because it's the more general structure and it gives me per-school estimates with proper uncertainty (a *modeling* reason, not a LOO reason), but I will not claim a predictive advantage I can't measure."

> ⚠️ **Pitfall — the cardinal sin.** Reporting "the hierarchical model wins, $\Delta$elpd $= 0.34$" as if 0.34 were a victory. It is **well inside one `dse`**. Selecting models on sub-`dse` differences is how you build a career of irreproducible "improvements." When the difference is small, *say so*, and choose on other grounds (interpretability, the question you actually need answered, computational cost).

### Visualize it: `az.plot_compare`

```python
az.plot_compare(compare_df)
plt.show()
```

> 🩺 **Diagnostic — reading the `plot_compare` figure.** Each model is a row (best on top if `order_by_rank=True`). The **open circle** is `elpd_loo` with a grey error bar of length `se`. For every model below the best, a **triangle with a bar** shows `elpd_diff` and its `dse` — *this* is the one to eyeball: if that bar comfortably overlaps the best model's vertical line, the difference is noise (our case). A filled marker appears only if you set `insample_dev=True`, showing in-sample deviance so you can *see* the optimism gap between in-sample and LOO. When the two rows' error bars overlap as heavily as ours do, the picture says "indistinguishable" at a glance — which is the whole point of plotting it.

> 🔧 **In practice — verify defaults.** `az.plot_compare`'s defaults (`insample_dev`, `plot_ic_diff`, `order_by_rank`) have flipped across ArviZ versions. Don't assert them from memory — `help(az.plot_compare)` in your installed version. The verified signature: `az.plot_compare(comp_df, insample_dev=False, plot_standard_error=True, plot_ic_diff=True, order_by_rank=True, legend=True, title=True, ...)`.

---

## 8. Don't pick — *average*: stacking and pseudo-BMA+

Here's a question that should bother you. We just decided two models were indistinguishable. So which do we deploy? "Pick the one with rank 0" throws away a perfectly good model and pretends to a certainty we don't have. The grown-up move, when models are close (or even when one is somewhat better but the other captures something it misses), is **not to pick at all** — it's to *average their predictive distributions*. That `weight` column in `az.compare` was the answer all along.

Model averaging combines the candidates into one mixture predictive distribution:

$$
p(\tilde y \mid y) \;=\; \sum_{k=1}^{K} w_k \; p(\tilde y \mid y, M_k), \qquad w_k \ge 0,\ \sum_k w_k = 1 .
$$

The only question is **how to choose the weights $w_k$**, and there are two good answers and one bad one.

### Stacking (the default, and usually the right choice)

**Stacking** chooses the weights on the simplex that *directly maximize the LOO log score of the mixture*:

$$
\hat w \;=\; \arg\max_{w \in \Delta}\ \sum_{i=1}^n \log \sum_{k=1}^K w_k\, p(y_i \mid y_{-i}, M_k).
$$

Read that carefully: it doesn't score each model and then combine the scores — it asks "what blend of these models, *combined*, would have predicted the held-out points best?" That's a fundamentally better question, because two mediocre models that err in *different* directions can blend into an excellent one (just like an ensemble in ML). Stacking is the ArviZ default (`method="stacking"`).

> 💡 **Intuition — the M-open world.** Stacking is justified for the **M-open** setting: the true data-generating process is **not** in your set of models (which is *always* the case in practice — "all models are wrong"). In M-open, the goal isn't to find the "true" model's probability; it's to build the best *predictive combination* from the wrong models you have. Stacking is provably the right tool for that goal. By contrast, classical **Bayesian model averaging** with marginal-likelihood weights assumes **M-closed** (the truth *is* one of your models) and inherits all the Bayes-factor pathologies of §11 — so we do **not** use BMA weights here.

### Pseudo-BMA+ (the cheaper alternative)

**Pseudo-BMA+** computes Akaike-style weights from the `elpd_loo` values — essentially $w_k \propto \exp(\mathrm{elpd}_{\mathrm{loo},k})$ — but with a **Bayesian-bootstrap** correction that accounts for the uncertainty in each `elpd_loo` (so a model that's only better by noise doesn't grab all the weight). It's cheaper than stacking and more stable than naive Akaike weights, but it generally produces **softer, less optimal** weights than stacking because it scores each model in isolation rather than as a blend.

```python
# Stacking weights (default) vs pseudo-BMA+ (Bayesian-bootstrap) weights
stack_df = az.compare({"hierarchical": idata_hier, "pooled": idata_pooled},
                      method="stacking")
bbpb_df  = az.compare({"hierarchical": idata_hier, "pooled": idata_pooled},
                      method="BB-pseudo-BMA")   # this IS pseudo-BMA+ (Bayesian bootstrap)

print("stacking weights:   ", stack_df["weight"].to_dict())
print("pseudo-BMA+ weights:", bbpb_df["weight"].to_dict())
```

> 🔧 **In practice — the `method` argument only changes `weight`.** `method` ∈ `{"stacking"` (default), `"BB-pseudo-BMA"` (= pseudo-BMA+), `"pseudo-BMA"` (no bootstrap — discouraged, too overconfident)`}`. It affects **nothing but the `weight` column** — `elpd_loo`, `elpd_diff`, and `dse` are identical across methods. So compute the table once to read the ELPDs, and switch `method` only to see how the averaging weights move.

> ⚠️ **Pitfall — a `weight` of 0.0 next to a small `elpd_diff`.** You will sometimes see stacking assign weight **0** to a model whose `elpd_diff` is modest. That's not a bug. Stacking finds the *optimal blend*, and if model B is a near-strict-subset of model A's predictive behavior, the blend that maximizes the LOO score puts everything on A. A zero stacking weight means "this model adds nothing *given the others*," which is different from "this model is bad in isolation." If you want softer weights for reporting, look at `BB-pseudo-BMA`.

> 📜 **Citation/Origin:** Stacking of predictive distributions is **Yao, Vehtari, Simpson & Gelman (2018)**, "Using Stacking to Average Bayesian Predictive Distributions" (*Bayesian Analysis* 13(3)) — the source for the M-open argument and the demonstration that stacking beats pseudo-BMA+ and BMA for prediction. The `loo` package vignette "Bayesian Stacking and Pseudo-BMA weights" (linked in Resources) works the weights by hand.

### Predicting with the averaged model

To actually *use* the averaged model, sample posterior predictions from each model and mix them in proportion to the weights:

```python
# pseudocode — average posterior-predictive draws by the stacking weights
w = stack_df["weight"]                      # {"pooled": 0.78, "hierarchical": 0.22}, say
ppc_pooled = pm.sample_posterior_predictive(idata_pooled, model=pooled_model,
                                            extend_inferencedata=False)
ppc_hier   = pm.sample_posterior_predictive(idata_hier,   model=hier_model,
                                            extend_inferencedata=False)

# Concrete weighted concatenation: take n_k = round(w_k * N) draws from each model's
# posterior-predictive and stack them into ONE averaged predictive sample.
yp_pooled = ppc_pooled.posterior_predictive["y_obs"].stack(s=("chain", "draw")).values  # (J, S)
yp_hier   = ppc_hier.posterior_predictive["y_obs"].stack(s=("chain", "draw")).values     # (J, S)
N = yp_pooled.shape[1]
n_pool = int(round(w["pooled"] * N)); n_hier = N - n_pool
cols_pool = rng.choice(yp_pooled.shape[1], size=n_pool, replace=False)
cols_hier = rng.choice(yp_hier.shape[1],   size=n_hier, replace=False)
y_avg = np.concatenate([yp_pooled[:, cols_pool], yp_hier[:, cols_hier]], axis=1)  # (J, N)
# y_avg is now ONE averaged predictive sample: ~78% of its columns came from the pooled
# model, ~22% from the hierarchical one. Summarize it like any predictive draw —
# az.plot_dist / np.percentile(y_avg, [3, 97], axis=1) for per-school 94% intervals.
```

> 🔧 **In practice:** For a quick win, ArviZ's `az.weight_predictions` (newer versions) takes a list of InferenceData and a weight vector and returns a single averaged posterior-predictive InferenceData — the canonical call is `az.weight_predictions([idata_pooled, idata_hier], weights=[0.78, 0.22])` (verify it exists in your installed ArviZ); otherwise do the proportional concatenation above by hand. Either way the result is a single predictive distribution whose mixture is dominated by the higher-weight model but visibly *wider* wherever the two models disagree — that extra spread is the averaging honestly carrying the model uncertainty forward. The key idea: average the **predictive distributions**, not the parameters (the models' parameters may not even be comparable).

---

## 9. Calibration: LOO-PIT and posterior predictive checks

ELPD is a single number summarizing predictive accuracy. But a model can have a perfectly respectable `elpd_loo` and still be **miscalibrated** — systematically over- or under-confident, or biased in its mean. Numbers don't catch everything; you need the *shape* of the predictive fit too. Two complementary tools, both qualitative, both essential alongside the ELPD table.

### LOO-PIT: are the leave-one-out predictions calibrated?

The **probability integral transform (PIT)** has a magic property: if you push an observation through the CDF of its own correct predictive distribution, you get a value that is **Uniform(0,1)**. The leave-one-out version asks, for each point, *where in its own leave-one-out predictive distribution did the actual observation fall?*

$$
\mathrm{LOO\text{-}PIT}_i \;=\; \Pr\!\big(\tilde y_i \le y_i \mid y_{-i}\big).
$$

If the model is well-calibrated, these $n$ values should look like a uniform sample on $[0,1]$. Systematic departures from uniformity are a fingerprint of *how* the model is miscalibrated:

> 🩺 **Diagnostic — reading LOO-PIT shapes.**
> - **Uniform** (flat KDE / ECDF on the diagonal): calibrated. 🎉
> - **∪-shape** (mass piled at 0 and 1): the predictions are **too narrow / overconfident** — real observations keep landing in the tails the model thought were unlikely. This is *under-dispersion*, the classic signature of Poisson-when-you-needed-negative-binomial (Ch 09).
> - **∩-shape** (mass piled in the middle): predictions are **too wide / under-confident** (over-dispersed).
> - **Sloped or shifted**: the predictive **mean is biased** (systematically too high or too low).

```python
# LOO-PIT requires BOTH a posterior_predictive group AND a log_likelihood group
pm.sample_posterior_predictive(idata_hier, model=hier_model, extend_inferencedata=True)

az.plot_loo_pit(idata_hier, y="y_obs")              # KDE of PIT vs Uniform, with confidence band
az.plot_loo_pit(idata_hier, y="y_obs", ecdf=True)   # Δ-ECDF view — easier to read; PREFER this
```

The `ecdf=True` view plots the *difference* between the empirical CDF of the PIT values and the uniform CDF, with a simultaneous confidence band; if the line stays inside the band, you're calibrated. It's easier to read than the KDE because "flat inside the band" is a cleaner target than "matches a uniform density." (Both need the `posterior_predictive` **and** `log_likelihood` groups — compute log-likelihood as in §6.)

> 🔧 **In practice:** LOO-PIT is the calibration check that the plain posterior predictive check *misses*, because it cross-validates — it asks about held-out calibration, not in-sample. We met it first in the workflow chapter (Ch 08); here it earns its place as a model-*comparison* tool: of two models with similar `elpd_loo`, prefer the one whose LOO-PIT is flatter.

### Posterior predictive checks as a comparison tool

The qualitative twin of ELPD is the **posterior predictive check** (Ch 08): simulate replicated datasets from each fitted model and overlay them on the real data. A model can win on `elpd_loo` yet visibly fail to reproduce a feature you care about — the proportion of zeros, the right tail, a bimodality.

```python
# Compare two models' posterior predictive fit side by side
pm.sample_posterior_predictive(idata_pooled, model=pooled_model, extend_inferencedata=True)
# idata_hier already has its PPC from above

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
az.plot_ppc(idata_pooled, ax=axes[0]); axes[0].set_title("pooled")
az.plot_ppc(idata_hier,   ax=axes[1]); axes[1].set_title("hierarchical")
plt.show()
```

> 🩺 **Diagnostic — `az.plot_ppc`.** The thick black line is the observed data's KDE; the thin coloured lines are KDEs of datasets simulated from the posterior. You want the black line to sit comfortably inside the cloud of coloured lines. For *comparison*, look for **a specific feature one model captures and the other doesn't** — that's often more decisive than a 0.3-elpd difference, and it's the kind of thing you can explain to a stakeholder. For richer PPC checks (test statistics, Bayesian p-values via `az.plot_bpv`), see Ch 08.

> 💡 **Intuition — three lenses on one question.** `elpd_loo` (how accurate), LOO-PIT (how calibrated), and PPC (does it reproduce the features I care about) are three views of "is this a good predictive model?" A complete comparison reports all three. The number ranks them; the calibration and PPC plots tell you *whether to believe the ranking* and *what each model gets wrong*.

---

## 10. When PSIS-LOO fails: K-fold cross-validation by hand

When too many $\hat k$ exceed 0.7 — or when LOO simply isn't applicable (time-series, where leaving out a middle point is cheating, or grouped data where you want leave-one-*group*-out) — fall back to honest **K-fold CV**. ArviZ won't refit your model for you, so you write the fold loop. The recipe is always the same: split the data, refit on each training fold, accumulate the **held-out** pointwise log-likelihood, then hand the assembled log-likelihood to `az.compare`.

```python
# pseudocode — K-fold CV by hand (the universal fallback)
from sklearn.model_selection import KFold

def kfold_elpd(build_model, y_all, x_all, K=10, seed=RANDOM_SEED):
    kf = KFold(n_splits=K, shuffle=True, random_state=seed)
    pointwise_ll = np.zeros((/* n_chains*n_draws */, len(y_all)))  # filled per held-out point
    for train_idx, test_idx in kf.split(y_all):
        # 1. fit on the TRAINING fold only
        model_train = build_model(y_all[train_idx], x_all[train_idx])
        with model_train:
            idata_k = pm.sample(1000, tune=1000, target_accept=0.9,
                                random_seed=seed, progressbar=False)
        # 2. score the HELD-OUT fold under that posterior (swap data via pm.Data / a fresh
        #    model on x_all[test_idx], then pm.compute_log_likelihood on the test points)
        # 3. write those columns into pointwise_ll[:, test_idx]
    # 4. sum the held-out log predictive densities over points -> elpd_kfold
    return pointwise_ll

# Assemble the held-out log-likelihood into an InferenceData's log_likelihood group,
# then az.compare({...}) exactly as with LOO. The loo R package's kfold() is the reference impl.
```

> 🔧 **What to expect from the number.** `elpd_kfold` plays the exact same role as `elpd_loo` — same scale, higher = better, and you read `elpd_diff`/`dse` between models identically. When you reach for K-fold *because* LOO failed (high $\hat k$ on the influential points), expect `elpd_kfold` to come out **slightly lower (more pessimistic) than the PSIS-LOO value it replaces**: PSIS was being optimistically biased on exactly those hard points, so the honest refit pulls the estimate down. If the two models' *ordering* flips when you move from LOO to K-fold, trust K-fold — that flip is the high-$\hat k$ warning made concrete.

> 🔧 **In practice:** K-fold is more expensive (K refits) but **always valid** — no importance-sampling assumption to break. Use it as the tie-breaker when $\hat k$ is red, and as the *only* honest option for dependent data (use blocked/forward-chaining folds for time series, Ch 15). For leave-one-group-out (the hierarchical analogue), make each group a fold — this answers "how well do I predict a *new school*?" rather than "a new student in a known school," which is often the question you actually care about.

---

## 11. The trouble with Bayes factors

Someone will eventually ask you for a **Bayes factor** — "just tell me the odds in favor of model 1." You need to understand precisely why this is a loaded request, because Bayes factors *feel* like the natural Bayesian answer to model comparison and yet are, in most practical settings, a trap. Let me be fair to them and then show you the landmines.

A Bayes factor is the ratio of the two models' **marginal likelihoods** (a.k.a. *evidence*):

$$
\mathrm{BF}_{12} \;=\; \frac{p(y \mid M_1)}{p(y \mid M_2)}, \qquad
p(y \mid M) \;=\; \int p(y \mid \theta)\, p(\theta)\, d\theta .
$$

Stare at that marginal likelihood. It is the likelihood **averaged over the prior** — not the posterior, the *prior*. That single fact is the source of every problem.

> ⚠️ **Pitfall 1 — extreme prior sensitivity.** Because $p(y \mid M)$ averages over the prior, the Bayes factor is exquisitely sensitive to prior *width* on parameters that exist only under one model. Suppose $M_1$ has an extra parameter $\beta$ with prior $\mathrm{Normal}(0, \sigma_\beta)$. Widen $\sigma_\beta$ — say from 10 to 1000, a change that barely touches the **posterior** and barely touches `elpd_loo` — and you spread the prior mass over a vast region where the likelihood is tiny, dragging $p(y \mid M_1)$ down and pushing the BF toward $M_2$, **regardless of what the data say**. You can move a Bayes factor by orders of magnitude with a prior change that a posterior-predictive analysis wouldn't even notice. A quantity that swings wildly on a choice you made before seeing the data is not a quantity you want to ship a decision on.

> ⚠️ **Pitfall 2 — the Jeffreys–Lindley paradox.** Take a fixed, *just-barely-significant* effect and let the prior under the alternative get more diffuse (or let $n \to \infty$). The Bayes factor will favor the **null** arbitrarily strongly — even as a frequentist test screams "significant!" The two frameworks can flatly contradict each other on the same data. The culprit is again the prior-averaging: a diffuse alternative "wastes" prior mass, and the evidence penalizes it. This isn't a bug to be patched; it's intrinsic to evidence-based testing, and it means a Bayes factor's verdict is only as meaningful as the *deliberate, informative* prior you put on the alternative.

> ⚠️ **Pitfall 3 — improper priors break it entirely.** If you use an improper prior (flat over the whole real line) — fine for estimation, since the posterior can still be proper — the marginal likelihood contains an **arbitrary normalizing constant**, so the Bayes factor is literally **undefined**. The vague "non-informative" priors people reach for to "let the data speak" make Bayes factors meaningless.

Contrast all of this with **LOO/ELPD**, which conditions on the data through the **posterior** predictive distribution and is therefore far less prior-sensitive: once the data have spoken, a moderately wider prior changes the posterior, and hence `elpd_loo`, only slightly. This robustness is the deep reason this course makes LOO the default and treats Bayes factors as a specialist tool.

### When *are* Bayes factors legitimate?

Not never — but the conditions are demanding. Bayes factors answer a genuinely different question than LOO: **LOO asks "which model predicts better?"; the Bayes factor asks "which model is more probable to have *generated* the data?"** That latter question is well-posed only in the **M-closed** world (one of your models really is the truth) *and* only when you can put **proper, carefully elicited, informative priors** on every parameter — because the BF's whole job is to integrate the likelihood against those priors, so the priors *are* part of the hypothesis. Genuine, pre-registered hypothesis testing in confirmatory science (a specific physical theory vs its negation, with priors elicited from theory) is the legitimate home of the Bayes factor.

> 🔧 **In practice — computing a marginal likelihood (if you must).** The marginal likelihood is hard to estimate: the notorious **harmonic-mean estimator** is so unstable Radford Neal called it "the worst Monte Carlo method ever." The gold standard is **bridge sampling** (the `bridgesampling`/`bridgestan` lineage; Gronau et al. 2017). In PyMC, **`pm.sample_smc`** (Sequential Monte Carlo) returns a log marginal likelihood as a byproduct in its `sample_stats` — convenient, but verify the exact key name in your installed version, and prefer bridge sampling when the decision is consequential.

```python
# pseudocode — marginal likelihood via SMC (byproduct), for a Bayes factor if you truly need one
with model_1:
    idata_smc_1 = pm.sample_smc(random_seed=RANDOM_SEED)
# the log marginal likelihood is reported in idata_smc_1.sample_stats (verify the key name);
# BF_12 = exp(logml_1 - logml_2).  Then RE-RUN with a wider prior and watch BF move — that
# sensitivity check is mandatory before you report any Bayes factor.
```

> 💡 **The bottom line.** For "which model should I use to predict / make decisions?" — the question 95% of practitioners actually have — use **LOO/ELPD and stacking**, not Bayes factors. Reach for a Bayes factor only for deliberate M-closed hypothesis testing with proper informative priors, compute it with bridge sampling, and *always* report a prior-sensitivity analysis alongside it. If a Bayes factor flips when you widen a prior, that's not a deeper truth — it's the method telling you it can't help you here.

---

## 12. A workflow for comparison (the checklist)

Pulling it together — when you face "which model?", run this loop:

1. **Converge every candidate first.** $\hat R \le 1.01$, ESS $\gtrsim 400$, zero divergences (Ch 07) on *each* model. An unconverged model has no valid LOO.
2. **Compute pointwise log-likelihood** (`idata_kwargs={"log_likelihood": True}` or `pm.compute_log_likelihood`).
3. **`az.loo(..., pointwise=True)` per model**; read the **Pareto-$k$ table**. Any $\hat k \ge 0.7$? Identify, investigate, and fix (better model / moment-match / K-fold) before trusting the number.
4. **`az.compare({...}, ic="loo")`**; read `elpd_diff` against `dse`. If $|{\rm elpd\_diff}| < 4$ → **no meaningful difference**; don't select on it.
5. **`az.plot_compare`** to *see* the overlap; **LOO-PIT** and **PPC** for calibration and feature-fit.
6. **Don't pick if they're close — stack.** Report stacking weights; average predictive distributions for deployment.
7. **Bayes factors:** only for M-closed testing with proper informative priors + bridge sampling + a mandatory prior-sensitivity check. Otherwise, no.

Print this in your head. It's the difference between "the bigger model has higher $R^2$, ship it" and a defensible decision.

---

## ⚠️ Common errors & how to fix them

| Symptom | Cause | Fix |
|---|---|---|
| `az.loo` / `az.waic` raises `TypeError`/`KeyError`: "log_likelihood not found in InferenceData" | PyMC v5 doesn't store pointwise log-likelihood by default | Add `idata_kwargs={"log_likelihood": True}` to `pm.sample`, **or** `with model: pm.compute_log_likelihood(idata)` afterward |
| `log_likelihood` group missing even after computing it | Likelihood was a bare `pm.Potential`, which `compute_log_likelihood` can't score | Express the likelihood as an **observed** RV (`pm.Dist("y", ..., observed=y)`); for custom densities use `pm.CustomDist` with an explicit `logp`, not `Potential` |
| ArviZ warns "Estimated shape parameter of Pareto distribution is greater than 0.7" | Heavy-tailed LOO importance ratios → influential/outlier points | Identify the points; try a more robust model (e.g., Student-$t$), PSIS **moment matching** (R `loo`'s `loo_moment_match()`; no canonical `az` call yet), an exact **refit of just the flagged points** ("reloo" — hand-rolled as a small fold loop, §4/§10), or full **K-fold**; report which observations |
| `az.compare` errors about mismatched `n_data_points` / observations | Models fit on **different data**, a transformed `y`, or a different observed variable | Compare only models with the **same observations on the same scale**; pass `var_name=` if there are multiple observed RVs |
| Comparing a model on `log(y)` vs one on `y` and declaring a winner | ELPD is on **different scales**; a change-of-variables Jacobian is missing | Put both likelihoods on the *same* response; if you transform `y`, add the Jacobian so the densities are comparable |
| Over-trusting a model with `elpd_diff = 0.3` as the "winner" | `elpd_diff < 4` and within one `dse` — it's noise | Report "no meaningful difference"; choose on interpretability / the question, or **stack** |
| WAIC and LOO disagree noticeably | WAIC `p_waic` unstable (var > 0.4 points), small $n$, or flexible model | Trust **LOO**; investigate high-$\hat k$ / high-`p_waic` points (usually the same); consider K-fold |
| `weight = 0.0` for a model whose `elpd_diff` is small | Stacking put all weight on a near-equivalent better model | Expected — it means "adds nothing given the others"; inspect `method="BB-pseudo-BMA"` for softer weights |
| A Bayes factor flips when you widen a prior | The marginal likelihood is a **prior average**; this is intrinsic prior sensitivity | Don't use BFs for model *choice*; use **LOO/stacking**. If you genuinely need a BF, fix proper informative priors, use **bridge sampling**, and report the sensitivity |
| `p_loo` is large (≈ $n$) with many high $\hat k$ | Severe misspecification or a very flexible model | If a PPC also fails → **fix the model**; if it's an honestly flexible model → **K-fold** |
| LOO numbers look fine but the model is clearly wrong on a feature | ELPD is a scalar; it can miss specific calibration failures | Always pair ELPD with **LOO-PIT** (∪ = overconfident, ∩ = underconfident) and a **PPC** |

---

## 🧪 Exercises

**1. (Conceptual) Why in-sample fit cannot select models.** Re-run the polynomial overfitting demo from §1, but this time *also* generate a fresh test set `y_test` from the same `true_mu` and compute the **test** $R^2$ for each degree. Plot in-sample and test $R^2$ on the same axes. Explain in two sentences why the curves diverge, and identify the degree that minimizes test error. *Hint: the test curve should be U-shaped (in error) — the bottom is the bias–variance sweet spot; in-sample is monotone.*

**2. (Coding) The third model.** Add a **no-pooling** model to the Eight Schools comparison: each $\theta_j \sim \mathrm{Normal}(0, 10)$ independently, no shared $\tau$. Compute its LOO, add it to `az.compare`, and read the table. Does no-pooling's `p_loo` go *up* toward 8? Does its `elpd_loo` drop? Explain what `p_loo` is telling you about overfitting, and connect it to the shrinkage story from Ch 10. *Hint: with no shared $\tau$, each $\theta_j$ is fit by essentially one data point — maximum overfitting, $p_{\text{loo}}$ near the parameter count.*

**3. (Coding + interpretation) Poisson vs Negative Binomial.** Take a count dataset (the **bike-sharing** or **roaches** data from Ch 09, or simulate over-dispersed counts with `rng.negative_binomial`). Fit a Poisson GLM and a Negative-Binomial GLM, compute LOO for both, run `az.compare`, and — crucially — plot **LOO-PIT** for each. You should find the NB wins on `elpd_loo` *and* that Poisson's LOO-PIT is **∪-shaped** (overconfident, because Poisson forces variance = mean). Write one paragraph connecting the ∪-shape to the over-dispersion you saw in the Ch 09 PPC. *Hint: this is the cleanest real demonstration in the chapter that LOO-PIT diagnoses the *kind* of failure, not just its presence.*

**4. (Diagnostic) Manufacture a high $\hat k$.** Add a gross outlier to the Howell !Kung height data (Ch 03) — e.g., a 250 cm "person" — and fit a Normal linear regression. Run `az.loo(idata, pointwise=True)` and find the outlier's $\hat k$ (it should be high). Now refit with a **Student-$t$** likelihood and re-check $\hat k$. Explain why the robust model's $\hat k$ for that point is lower, in terms of how surprised each model is by the outlier. *Hint: $\hat k$ measures how much leaving the point out would move the posterior — a $t$-likelihood barely flinches at the outlier, so the point is far less influential.*

**5. (Conceptual, hard) Break a Bayes factor on purpose.** For a simple regression with one optional predictor $\beta$, compute (or reason about) the Bayes factor of "with $\beta$" vs "without," under priors $\beta \sim \mathrm{Normal}(0, \sigma_\beta)$ for $\sigma_\beta \in \{1, 10, 100, 1000\}$. Show that the BF moves substantially across these while the **posterior** for $\beta$ and the **`elpd_loo`** barely move. Write the moral in your own words. *Hint: this is the Jeffreys–Lindley mechanism in miniature — the marginal likelihood averages over the prior, so wider prior = more wasted mass = lower evidence for the bigger model.*

**6. (Applied judgment) Stacking decision.** For the Eight Schools comparison, compute weights under both `method="stacking"` and `method="BB-pseudo-BMA"`. Suppose a stakeholder demands a *single* model. Write the two-sentence recommendation you'd actually send, citing `elpd_diff`, `dse`, and the weights — and defend choosing the hierarchical model on *modeling* grounds even though it doesn't win on LOO. *Hint: "predictively indistinguishable; I keep the hierarchical model for the per-group estimates and honest uncertainty it provides, not for a predictive edge it doesn't have."*

---

## 📚 Resources & further reading

**The papers that define the methods.**
- **Vehtari, Gelman & Gabry (2017), "Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC," *Stat. Comput.* 27:1413–1432** — the reference for ELPD, LOO, WAIC, and PSIS. §2–3 give every formula in this chapter. https://link.springer.com/article/10.1007/s11222-016-9696-4 (preprint: https://arxiv.org/abs/1507.04544).
- **Vehtari, Simpson, Gelman, Yao & Gabry (2024), "Pareto Smoothed Importance Sampling," *JMLR*** — defines PSIS and the sample-size-adaptive $\hat k$ threshold (`good_k`). https://arxiv.org/abs/1507.02646.
- **Yao, Vehtari, Simpson & Gelman (2018), "Using Stacking to Average Bayesian Predictive Distributions," *Bayesian Analysis* 13(3):917–1007** — stacking vs pseudo-BMA+, the M-open argument. https://doi.org/10.1214/17-BA1091 (preprint: https://arxiv.org/abs/1704.02030).
- **Gneiting & Raftery (2007), "Strictly Proper Scoring Rules, Prediction, and Estimation," *JASA*** — why the log score is the principled target.
- **Gronau, Singmann & Wagenmakers (2017), "bridgesampling," arXiv:1710.08162** — bridge sampling for marginal likelihoods / Bayes factors, and why naive (harmonic-mean) estimators fail. https://arxiv.org/abs/1710.08162.

**The practical bibles.**
- **Vehtari, "Cross-validation FAQ"** — https://users.aalto.fi/~ave/CV-FAQ.html — the operational reference: the "`elpd_diff < 4` is small, otherwise compare to SE" rule, when the SE is unreliable, when to use K-fold, LOO-vs-WAIC guidance. Read it cover to cover.
- **PyMC "Model comparison" core notebook** — https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/model_comparison.html — the verbatim `idata_kwargs={"log_likelihood": True}` / `pm.compute_log_likelihood` / `az.loo`/`az.compare`/`az.plot_compare` workflow on Eight Schools, with the table-interpretation paragraphs.
- **loo R package — Pareto-$k$ diagnostic** — https://mc-stan.org/loo/reference/pareto-k-diagnostic.html — exact threshold language and the $k \ge 1$ non-finite-mean caveat.
- **loo R package — glossary** — https://mc-stan.org/loo/reference/loo-glossary.html — the definitive `p_loo` / $\hat k$ readings (misspecification vs flexible model).
- **loo vignette — "Bayesian Stacking and Pseudo-BMA weights"** — https://mc-stan.org/loo/articles/loo2-weights.html — stacking vs pseudo-BMA+ weights worked by hand.

**ArviZ API docs (verify signatures against your installed version).**
- `arviz.loo` — https://python.arviz.org/en/stable/api/generated/arviz.loo.html
- `arviz.compare` — https://python.arviz.org/en/stable/api/generated/arviz.compare.html
- `arviz.plot_compare` — https://python.arviz.org/en/stable/api/generated/arviz.plot_compare.html
- LOO-PIT ECDF example — https://python.arviz.org/en/stable/examples/plot_loo_pit_ecdf.html

**Narrative companions.**
- **McElreath, *Statistical Rethinking* (2nd ed., 2020), Ch. 7 "Ulysses' Compass"** — overfitting, information theory, regularization, WAIC/PSIS — the best *story* of why we do any of this. PyMC port: https://github.com/pymc-devs/pymc-resources.
- **Martin, Kumar & Lao, *Bayesian Modeling and Computation in Python*, Ch. 2 "Exploratory Analysis of Bayesian Models"** — https://bayesiancomputationbook.com/markdown/chp_02.html — the course's companion treatment of LOO, `az.compare`, LOO-PIT, and PPCs in idiomatic ArviZ.
- **Gelman et al., "Bayesian Workflow" (2020), arXiv:2011.01808** — situates comparison inside the larger iterative loop (Ch 08).

---

## ➡️ What's next

You can now build models, diagnose them, check them, and — as of this chapter — **compare and average them honestly**, scoring on out-of-sample prediction instead of in-sample flattery, and refusing to over-read a difference that's smaller than its own standard error. That completes the core toolkit of the Bayesian workflow.

The next several chapters return to *building* — now with much richer models. **Chapter 12 (Gaussian Processes)** opens the door to nonparametric regression: instead of choosing a fixed functional form (linear? quadratic? which spline?), you put a prior directly on *functions* and let the data choose the shape, with calibrated uncertainty everywhere. And you'll find LOO waiting for you there too — comparing kernels and lengthscale priors is exactly the model-comparison problem you just learned to solve, with $\hat k$ keeping you honest about the influential points a flexible GP will inevitably strain against.
