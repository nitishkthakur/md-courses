# Chapter 01 — Foundations of Bayesian Inference

### Probability as extended logic, Bayes' theorem from the ground up, and the generative mindset that powers everything that follows

> _A note from me to you._
>
> _Before you fit a single model in PyMC, I want you to feel the engine that makes it run.
> Almost every confusing thing you will ever hit in Bayesian practice — a credible interval that
> a reviewer "corrects" into something wrong, a posterior that refuses to budge from a stubborn
> prior, a model that samples beautifully and yet predicts nonsense — traces back to a foundation
> that was learned as a formula instead of as a way of thinking. So in this chapter we go slow.
> We will derive Bayes' theorem rather than quote it, we will compute a posterior by hand with
> real numbers until the answer surprises you, and we will fit the same tiny problem two
> completely different ways (analytic conjugacy and a brute-force grid) so you can watch belief
> update one observation at a time. The single most expensive mistake a strong ML practitioner
> makes when they come to Bayes is treating it as "frequentist statistics with extra steps." It
> is not. It is a different definition of what a probability *is*. Get that right here, and the
> other seventeen chapters will feel like consequences. Get it wrong, and you will spend a year
> confused. Let's get it right._

---

## What you'll be able to do after this chapter

- Explain **probability as extended logic** (Cox/Jaynes) and articulate the two interpretations of probability — long-run frequency vs degree of belief — and why the Bayesian one is what licenses the statements we actually want to make.
- State the **sum rule** and **product rule** of probability and *derive* Bayes' theorem from them in two lines.
- Name the **four pieces** of Bayes' theorem — prior, likelihood, evidence/marginal, posterior — and say precisely what each one *does* and what it is a function *of*.
- Work a **discrete posterior by hand** (a medical-test base-rate problem) and explain why the base rate dominates the answer — the single most important intuition for diagnostic reasoning.
- Fit the **globe-tossing / coin problem two ways**: analytically via **Beta-Binomial conjugacy**, and numerically via **grid approximation**, with NumPy code you can run today, and watch the posterior update observation by observation.
- Distinguish **likelihood from probability** (what is held fixed, what varies) and stop confusing the two forever.
- Adopt the **generative modeling mindset**: a model is a recipe to *simulate* data; you run it forward to imagine, and condition it to learn.
- State honestly how **Bayesian and frequentist** inference differ — sampling distributions, confidence vs credible intervals and their genuinely different interpretations, p-values, when they agree and when the difference bites.
- Hold McElreath's **small-world / large-world** distinction in your head, which is the reason model checking exists at all.

---

## 1. Two meanings of one word

Sit with this for a second, because everything hinges on it. When you say "the probability is 0.7," you could mean one of two quite different things.

**Interpretation 1 — long-run frequency.** Probability is the limiting relative frequency of an event in an indefinitely repeated experiment. "This coin has probability 0.5 of heads" means: if I flip it forever, the fraction of heads tends to 0.5. Under this view, probability is a *physical property of a repeatable setup*. It is objective and it is empirical — but it is only defined for things you can imagine repeating. This is the **frequentist** interpretation, and it is the water most ML practitioners have been swimming in.

**Interpretation 2 — degree of belief (plausibility).** Probability is a number, between 0 and 1, that measures how strongly a rational agent should believe a proposition given the information it has. "The probability that it rained in Lagos yesterday is 0.7" is a perfectly sensible sentence under this view — even though yesterday is not repeatable and either it rained or it didn't. Probability here is *epistemic*: it lives in the relationship between evidence and a claim, not in the physics of a coin. This is the **Bayesian** interpretation.

> 💡 **Intuition:** The frequentist asks "how often?" The Bayesian asks "how much should I believe?" Both are useful questions. But notice: almost every question you actually care about in practice — *is this drug effective? what is this customer's churn risk? did this model improve the metric?* — is a question about a one-time, non-repeatable state of the world. The degree-of-belief interpretation answers exactly that question. The frequency interpretation has to answer a *different*, repeatable question and then hope you don't notice the swap. That swap is the source of a hundred misinterpreted p-values and confidence intervals.

Here is the part that elevates this from a philosophical preference to mathematics: **if you accept a few minimal requirements for a calculus of plausible reasoning, the rules of probability are forced on you.** You do not get to invent your own. This is Cox's theorem.

> 📜 **Citation/Origin:** R. T. Cox (1946) and, definitively, E. T. Jaynes, _Probability Theory: The Logic of Science_ (2003), chapters 1–2. Jaynes is the canonical, gloriously opinionated source for the view that probability theory **is** extended logic.

### Cox's theorem, in plain words

Suppose you want to build a machine that reasons about uncertain propositions. You insist on three desiderata:

1. **Plausibilities are represented by real numbers.** Greater plausibility = larger number. (You need *some* way to compare; reals are the natural choice.)
2. **Qualitative agreement with common sense / Boolean logic.** If new evidence makes $A$ more plausible, it shouldn't simultaneously make $A$ less plausible. In the limit of certainty, your calculus must reduce to ordinary deductive logic (true/false, AND/OR/NOT).
3. **Internal consistency.** If a plausibility can be computed two different valid ways, both ways must give the same answer. And you must use all the evidence you have, never arbitrarily ignoring some.

Cox proved that **any** system satisfying these is *isomorphic to probability theory* — it must obey the sum and product rules (below), up to a rescaling. There is no second, inequivalent calculus of consistent plausible reasoning hiding out there. Probability is not *a* way to reason under uncertainty; it is, in this precise sense, *the* way.

> 🧮 **The math:** "Isomorphic to probability theory" means: there exists a monotonic re-mapping of your plausibility numbers onto $[0,1]$ under which they satisfy
> $$ p(A) + p(\lnot A) = 1 \qquad\text{(sum rule)} \qquad\qquad p(A, B) = p(A \mid B)\, p(B) \quad\text{(product rule)}. $$
> Everything else — Bayes' theorem, marginalization, the whole apparatus — is algebra on top of these two.

This is the philosophical bedrock of "why Bayesian," and it's worth saying out loud: when a Bayesian writes $p(\theta)$ — a probability distribution over a *parameter*, over an *unknown quantity that has one true fixed value* — they are not committing a category error. They are using probability in its degree-of-belief sense, which Cox/Jaynes show is the unique consistent extension of logic to uncertainty. The frequentist objection "but $\theta$ isn't random!" is answered: correct, $\theta$ isn't random, but **our knowledge of it is uncertain**, and probability is the calculus of uncertainty, not just of randomness.

---

## 2. The two rules everything is built from

You only ever need two rules. Memorize them; you will use them for the rest of your life.

**The sum rule.** For a proposition $A$ and its negation:
$$ p(A) + p(\lnot A) = 1. $$
More generally, to get the probability of $A$ alone when $A$ is entangled with some other variable $B$, you **sum (or integrate) $B$ out** — this is *marginalization*, and it is the single most important computational move in Bayesian statistics:
$$ p(A) = \sum_{b} p(A, B=b) \qquad\text{or, continuous,}\qquad p(A) = \int p(A, B)\, dB. $$

**The product rule.** The joint probability of $A$ and $B$ factorizes into a conditional times a marginal:
$$ p(A, B) = p(A \mid B)\, p(B) = p(B \mid A)\, p(A). $$

ASCII, for skimmers:

```
sum rule:      p(A) = Σ_b p(A, B=b)          (marginalize: sum the nuisance out)
product rule:  p(A, B) = p(A|B) p(B) = p(B|A) p(A)
```

> 💡 **Intuition:** The product rule says "the chance of both is the chance of one, times the chance of the other *given* the first." The sum rule says "to forget about a variable you don't care about, add up all the ways it could have turned out." That's it. That's the whole grammar of probability. Bayes' theorem is one sentence written in this grammar.

### Deriving Bayes' theorem (two lines, do it yourself)

The product rule, written two ways, must agree:
$$ p(\theta, y) = p(y \mid \theta)\, p(\theta) = p(\theta \mid y)\, p(y). $$
Now just divide the right-hand equality by $p(y)$:
$$ \boxed{\; p(\theta \mid y) = \frac{p(y \mid \theta)\, p(\theta)}{p(y)} \;} $$

That is **Bayes' theorem**. It is not a deep, mysterious result; it is the product rule rearranged. The depth is entirely in the *interpretation* of the four pieces, which we turn to now.

---

## 3. The four named pieces, and what each one does

Write the theorem with names attached:

$$ \underbrace{p(\theta \mid y)}_{\text{posterior}} = \frac{\overbrace{p(y \mid \theta)}^{\text{likelihood}}\;\overbrace{p(\theta)}^{\text{prior}}}{\underbrace{p(y)}_{\text{evidence / marginal}}} $$

Here $\theta$ denotes the unknown parameter(s) we want to learn, and $y$ denotes the observed data. Let's take the four pieces one at a time, slowly, because the entire course is an elaboration of these four objects.

### The prior, $p(\theta)$

This is what you believe about $\theta$ **before** seeing this data — your state of knowledge going in. It is a genuine probability distribution over parameter values. It is *not* an embarrassing fudge factor to be minimized; it is information, and it does real work: it regularizes (pulls wild estimates toward sensible values), it encodes domain knowledge ("a human reaction time is not 4 hours"), and in hierarchical models it is the very mechanism that produces shrinkage and partial pooling (Chapter 10). We devote two entire chapters — **Chapter 02 (Priors I: Concepts and Common Priors)** and **Chapter 03 (Priors II: Choosing, Building, and Checking)** — to doing this well. For now: the prior is the question "what's plausible before the data speaks?"

### The likelihood, $p(y \mid \theta)$

This is your **generative assumption**: a probability model for the data *given* a specific value of the parameter. "If the true coin bias were $\theta = 0.3$, how probable is the exact sequence of heads and tails I saw?" The likelihood is where your modeling choices live — is the data Normal? Binomial? Poisson? Student-t? The likelihood is the bridge between parameters and observations.

> ⚠️ **Pitfall:** The likelihood $p(y \mid \theta)$ is **a function of $\theta$, not a probability distribution over $\theta$.** When you fix the data $y$ and let $\theta$ vary — which is exactly what you do during inference — the resulting function of $\theta$ does *not* integrate to 1. It is not a density over $\theta$. We will hammer this in §7, because confusing "likelihood" with "probability of the parameter" is the most common conceptual slip there is.

### The evidence / marginal likelihood, $p(y)$

Also called the **marginal likelihood** or just **the evidence**. By the sum rule, it is the likelihood averaged over *all* prior parameter values:
$$ p(y) = \int p(y \mid \theta)\, p(\theta)\, d\theta. $$
Read it carefully: it is the probability of the data, marginalizing $\theta$ out entirely — "how probable was this dataset under my model, before I knew the parameter?" Two things to know about $p(y)$:

1. **For inference about $\theta$, it is just a normalizing constant.** It does not depend on $\theta$ (we integrated $\theta$ away). So it only rescales the posterior so that it integrates to 1. This is why you constantly see the *proportional* form, which is what you actually compute:
   $$ p(\theta \mid y) \;\propto\; p(y \mid \theta)\, p(\theta), \qquad\text{ASCII: } \;\texttt{posterior} \propto \texttt{likelihood} \times \texttt{prior}. $$
2. **It is the hard part.** That integral is, in general, intractable — high-dimensional, no closed form. *Avoiding* the explicit computation of $p(y)$ is the entire reason MCMC (Chapter 05) exists: clever algorithms let us draw samples from the posterior using only the unnormalized product $p(y \mid \theta)\,p(\theta)$, never needing $p(y)$. (The evidence does come back as a star player when we *compare* models in Chapter 11, and as the thing prior predictive checks sample from — more on that in §6.)

> 🧮 **The math (marginalization is the sum rule again):** $p(y) = \int p(y,\theta)\,d\theta = \int p(y\mid\theta)\,p(\theta)\,d\theta$. The first equality is the sum rule (integrate the nuisance $\theta$ out of the joint); the second is the product rule (factor the joint). Two rules, every time.

### The posterior, $p(\theta \mid y)$

This is the prize: your updated state of belief about $\theta$ **after** seeing the data. It is a full probability distribution, not a point estimate. From it you can read off anything you want — a point summary (mean, median, mode), an interval (the *credible* interval, §8), the probability that $\theta$ exceeds some threshold, the expected value of a decision. The posterior is the answer Bayesian inference returns, and it answers exactly the question you asked: *given what I knew and what I saw, what should I now believe about $\theta$?*

> 💡 **Intuition (the whole theorem in one breath):** Start with what you believed (prior). Weigh each possibility by how well it explains the data (likelihood). Renormalize so it's a proper distribution (divide by evidence). What comes out is what you should now believe (posterior). *Belief in, evidence applied, updated belief out.* That loop — and the fact that yesterday's posterior is tomorrow's prior — is the heartbeat of the whole subject.

---

## 4. A discrete posterior by hand: the base-rate problem

Theory becomes intuition the moment you compute a posterior with real numbers and the answer ambushes you. Let me give you the classic ambush. It is worth doing slowly, because the lesson — **the base rate dominates** — is the single most important piece of probabilistic reasoning a data scientist can own, and it is the reason naive readings of "test accuracy" are dangerous.

**The setup.** A disease affects 1 in 1,000 people: the *prevalence* (base rate) is $p(D) = 0.001$. There is a test for it with these properties:

- **Sensitivity** (true positive rate): if you have the disease, the test is positive 99% of the time. $p(+ \mid D) = 0.99$.
- **Specificity** (true negative rate): if you are healthy, the test is negative 99% of the time. So the **false positive rate** is $p(+ \mid \lnot D) = 0.01$.

You take the test. **It comes back positive.** What is the probability you actually have the disease? Most people — including most clinicians, in the studies — answer "about 99%." Let's compute it.

The unknown $\theta$ here is binary: either you have the disease ($D$) or you don't ($\lnot D$). The data $y$ is the single observation "test positive" ($+$). Bayes' theorem:

$$ p(D \mid +) = \frac{p(+ \mid D)\, p(D)}{p(+)}. $$

We have the numerator pieces. We need the evidence $p(+)$ — the total probability of seeing a positive test, from *either* source. Sum rule, splitting over the two ways a positive can happen (true positive from a sick person, false positive from a healthy person):

$$ p(+) = \underbrace{p(+ \mid D)\,p(D)}_{\text{true positives}} + \underbrace{p(+ \mid \lnot D)\,p(\lnot D)}_{\text{false positives}}. $$

Plug in real numbers:

```
true positives  = p(+|D)  p(D)   = 0.99 × 0.001 = 0.00099
false positives = p(+|¬D) p(¬D)  = 0.01 × 0.999 = 0.00999
evidence p(+)   = 0.00099 + 0.00999 = 0.01098
```

Now assemble the posterior:

$$ p(D \mid +) = \frac{0.00099}{0.01098} \approx 0.0902. $$

**About 9%.** Not 99%. Read that again. A test that is "99% accurate" in both directions, when it fires positive, leaves you with only a **9% chance** of actually being sick. If a strong, "obviously correct" answer of 99% was sitting in your head a moment ago, this is the feeling we are chasing — the moment the arithmetic overrides the intuition and rebuilds it.

### Why the base rate dominates

Here is the engine of the surprise. The disease is *rare*: only 1 in 1,000 people have it. So when we test a large population, the few genuinely sick people produce a trickle of true positives, but the enormous healthy majority — even with only a 1% false-positive rate — produces a *flood* of false positives. Make it concrete with 100,000 people:

| | Have disease ($D$) | Healthy ($\lnot D$) | Total |
|---|---:|---:|---:|
| **Test +** | 99 | 999 | 1,098 |
| **Test −** | 1 | 98,901 | 98,902 |
| **Total** | 100 | 99,900 | 100,000 |

Of the 1,098 people who test positive, only 99 are actually sick: $99 / 1098 \approx 0.090$. The false positives (999 of them) outnumber the true positives (99) by **ten to one**, purely because there were a thousand times more healthy people to draw them from. The test's quality never changed — the *base rate* changed the conclusion.

> 💡 **Intuition (base-rate neglect):** The likelihood ratio — how much *more* a positive favors $D$ over $\lnot D$ — is $0.99 / 0.01 = 99$. That's strong evidence! But evidence multiplies the prior *odds*, it doesn't replace them. Prior odds of disease were $0.001 / 0.999 \approx 1{:}999$. Multiply by the likelihood ratio 99 and you get posterior odds $99{:}999 \approx 1{:}10$, i.e. probability $\approx 1/11 \approx 0.091$. (That $0.091$ is the odds $99{:}999$ rounded to $1{:}10$; carrying the unrounded numerator and denominator gives $99/(99+999) = 0.0902$, the exact value in the table above — same answer, the gap is only rounding.) The base rate was so lopsided that even 99-to-1 evidence couldn't overturn it. **Strong evidence applied to a strong prior gives a moderate posterior.** This is not a bug; it is correct reasoning, and the whole reason we insist on writing the prior down.

> 🔧 **In practice:** This is *exactly* why you re-test after a positive screen, and why screening rare conditions in low-risk populations produces mostly false alarms. It is also why, when a stakeholder shows you a classifier with "99% accuracy" on a 0.1%-prevalence problem, you should immediately compute its **positive predictive value**, not its accuracy. The arithmetic above *is* that computation. The Bayesian frame makes base-rate reasoning automatic instead of an afterthought.

> 🧮 **The math (odds form of Bayes, the working version):** Dividing Bayes' theorem for $D$ by the same for $\lnot D$ cancels the evidence entirely:
> $$ \underbrace{\frac{p(D\mid +)}{p(\lnot D\mid +)}}_{\text{posterior odds}} = \underbrace{\frac{p(+\mid D)}{p(+\mid \lnot D)}}_{\text{likelihood ratio (Bayes factor)}} \times \underbrace{\frac{p(D)}{p(\lnot D)}}_{\text{prior odds}}. $$
> Posterior odds = likelihood ratio × prior odds. No evidence integral needed; it cancels. This is the cleanest possible statement of "evidence updates belief," and we'll meet the likelihood ratio again as the **Bayes factor** in Chapter 11.

---

## 5. The globe toss: one problem, two methods, and watching belief update

The base-rate problem had a *discrete* unknown (sick or healthy). Now let's do a *continuous* one, the example McElreath uses to open _Statistical Rethinking_ because it is the smallest complete Bayesian analysis there is. We will solve it twice — once with brute-force **grid approximation**, once with exact **Beta-Binomial conjugacy** — and the two answers will agree, which is the point: grid approximation is the conceptual skeleton, conjugacy is the closed-form shortcut for one special case, and (foreshadowing) MCMC in Chapter 05 is what we use when neither shortcut is available.

**The setup.** You hold a small globe of the Earth. You toss it, catch it, and record whether your right index finger landed on **W**ater or **L**and. You repeat. The unknown $\theta$ is the true proportion of the globe's surface that is water — a continuous number in $[0,1]$. (Swap "globe / water" for "coin / heads" and it's identical: $\theta$ is the bias.) Suppose your nine tosses came out:

```
W L W W W L W L W      →  6 water, 3 land   (n = 9, w = 6)
```

**The likelihood.** Given a fixed $\theta$, the number of waters $w$ in $n$ independent tosses is **Binomial**:
$$ p(w \mid \theta, n) = \binom{n}{w}\, \theta^{w}\,(1-\theta)^{\,n-w}. $$
For our data, $p(6 \mid \theta, 9) = \binom{9}{6}\,\theta^{6}(1-\theta)^{3}$. As a function of $\theta$ (data fixed), this is the likelihood — and remember, it does *not* integrate to 1 over $\theta$.

**The prior.** Start maximally honest about our ignorance: every proportion is equally plausible, $p(\theta) = 1$ on $[0,1]$. That is a Uniform$(0,1)$ prior, which is also the Beta$(1,1)$ distribution — keep that fact in your pocket, it pays off in a minute.

### Method A — grid approximation (the universal, no-math-required method)

The idea is almost insultingly simple, and that is its virtue: you can't do the posterior integral in closed form in general, so **don't**. Lay down a fine grid of candidate $\theta$ values, evaluate `prior × likelihood` at each one, and normalize the resulting heights so they sum to 1. That normalized vector *is* a discrete approximation to the posterior. No calculus. It works for any prior and any likelihood you can evaluate. It only fails when $\theta$ has more than a few dimensions (the grid blows up exponentially — the "curse of dimensionality" that motivates MCMC), but for one or two parameters it is the clearest possible window into what Bayes' theorem is doing.

Here is the full, PyMC-free NumPy code. Run it.

```python
import numpy as np
import matplotlib.pyplot as plt

RANDOM_SEED = 8927

# --- data: 9 tosses, 6 water -------------------------------------------------
n = 9        # number of tosses
w = 6        # number of waters observed

# --- 1. a grid of candidate theta values (the "small world" of possibilities)
grid_size = 200
theta_grid = np.linspace(0, 1, grid_size)        # 200 candidate proportions

# --- 2. the prior, evaluated on the grid (flat = every theta equally plausible)
prior = np.ones_like(theta_grid)                 # Uniform(0,1) == Beta(1,1)

# --- 3. the likelihood of the data at each candidate theta ------------------
#     Binomial pmf; the n-choose-w constant is the same for every theta,
#     so it cancels in normalization and we could drop it — keep it for clarity.
from scipy.stats import binom
likelihood = binom.pmf(w, n, theta_grid)         # p(w | theta, n) at each grid point

# --- 4. unnormalized posterior = prior * likelihood -------------------------
unnormalized = prior * likelihood

# --- 5. normalize so the posterior sums to 1 (this is the grid version of /p(y))
posterior = unnormalized / np.sum(unnormalized)

# --- read off summaries ------------------------------------------------------
post_mean = np.sum(theta_grid * posterior)       # E[theta | data]
post_mode = theta_grid[np.argmax(posterior)]     # MAP estimate
print(f"posterior mean  ≈ {post_mean:.3f}")      # ≈ 0.636
print(f"posterior mode  ≈ {post_mode:.3f}")      # ≈ 0.668  (nearest grid point to w/n = 2/3, the MLE under a flat prior)
```

> 🩺 **Diagnostic (how to read this):** `np.sum(unnormalized)` is the grid's estimate of the **evidence** $p(y) = \int p(y\mid\theta)\,p(\theta)\,d\theta$ — literally a Riemann sum of likelihood × prior over the grid. Dividing by it is the discrete stand-in for the integral in Bayes' theorem. Notice the posterior mode lands *near* $w/n = 6/9 = 0.6\overline{6}$ — the program actually prints $0.668$, because $2/3$ is not one of the 200 grid nodes and the grid resolves it to the nearest one ($0.66834$); refine the grid and it creeps toward $0.667$. Under a *flat* prior the most probable parameter value is exactly the maximum-likelihood estimate, because the prior contributes a constant and the posterior is just a renormalized likelihood. The posterior *mean*, 0.636, is pulled slightly toward 0.5: the flat-prior posterior is the Beta distribution we'll meet next, whose mean is $(w+1)/(n+2)$, not $w/n$. That gap between mode and mean is your first encounter with shrinkage.

Now the payoff of grid approximation: we can **watch belief update one toss at a time.** Feed the tosses in sequentially and replot the posterior after each observation. Each new toss multiplies the current posterior (which becomes the new prior — yesterday's posterior is today's prior) by the one-toss likelihood.

```python
tosses = ["W","L","W","W","W","L","W","L","W"]    # the 9 observations, in order

theta_grid = np.linspace(0, 1, 200)
post = np.ones_like(theta_grid)                    # start from the flat prior
post /= np.sum(post)

fig, ax = plt.subplots(figsize=(7, 4))
for i, t in enumerate(tosses):
    # one-toss likelihood: theta if Water, (1 - theta) if Land
    lik = theta_grid if t == "W" else (1 - theta_grid)
    post = post * lik                              # update: posterior ∝ prior × likelihood
    post /= np.sum(post)                            # renormalize
    ax.plot(theta_grid, post, alpha=0.35 + 0.07 * i,
            label=f"after {i+1} tosses ({''.join(tosses[:i+1])})")
ax.set_xlabel(r"$\theta$ (true proportion water)")
ax.set_ylabel("posterior density")
ax.legend(fontsize=7)
plt.show()

# the single number that summarizes the final curve — compare it across orderings:
print(f"final posterior mean ≈ {post @ theta_grid:.3f}")   # ≈ 0.636, identical for any order of `tosses`

> 🩺 **Diagnostic (what the plot shows):** You will see a sequence of curves that start as a flat line (total ignorance) and, with each toss, grow taller and narrower, sliding toward the data's signal around 0.6–0.7. A **W** shifts mass right; an **L** shifts mass left; and the curve *concentrates* as $n$ grows — uncertainty shrinks with data, automatically, with no extra machinery. This single animation is Bayesian learning in its entirety: each datum reshapes belief, and the order doesn't matter — try shuffling `tosses` and the printed final mean stays $0.636$ and the final curve is identical, because multiplication commutes. That order-independence is a deep property: the posterior depends on the data only through the *counts*, not the sequence — a first taste of the **likelihood principle** (§8).

### Method B — Beta-Binomial conjugacy (the exact closed form)

Grid approximation is general but approximate (its accuracy is limited by grid resolution) and it doesn't scale past a few dimensions. For this specific problem there is something better. Watch what happens when the prior is a **Beta** distribution. The Beta density is
$$ p(\theta \mid \alpha, \beta) = \frac{\theta^{\alpha - 1}(1-\theta)^{\beta - 1}}{B(\alpha, \beta)}, \qquad \theta \in [0,1], $$
where $B(\alpha,\beta)$ is just the normalizing constant. Multiply a Beta$(\alpha,\beta)$ prior by the Binomial likelihood $\theta^{w}(1-\theta)^{n-w}$ and collect exponents:
$$ p(\theta \mid w) \;\propto\; \underbrace{\theta^{w}(1-\theta)^{n-w}}_{\text{likelihood}} \cdot \underbrace{\theta^{\alpha-1}(1-\theta)^{\beta-1}}_{\text{prior}} = \theta^{(\alpha + w) - 1}\,(1-\theta)^{(\beta + n - w) - 1}. $$

Look at the right-hand side: it has the *exact algebraic shape of a Beta density again*, with updated parameters. So **the posterior is Beta$(\alpha + w,\; \beta + n - w)$** — no integral required. A prior whose family is preserved by the likelihood is called **conjugate**, and Beta is the conjugate prior for the Binomial likelihood.

> 🧮 **The math (why this is magic, not coincidence):** Conjugacy works because the Binomial likelihood, *as a function of $\theta$*, is itself proportional to a $\theta^{a}(1-\theta)^{b}$ form — the same kernel as Beta. Multiplying two such kernels just adds the exponents. The Beta parameters get a beautiful interpretation: $\alpha$ acts like "prior waters" and $\beta$ like "prior lands" — **pseudo-counts** of imaginary data you walked in with. The flat Beta$(1,1)$ prior is "1 imaginary water, 1 imaginary land," i.e. the gentlest possible nudge. Updating literally means: add your real water count to $\alpha$, your real land count to $\beta$.

For our data ($w=6$ waters, $n-w=3$ lands) starting from the flat Beta$(1,1)$:
$$ \text{posterior} = \mathrm{Beta}(1 + 6,\; 1 + 3) = \mathrm{Beta}(7,\, 4). $$

And a Beta$(\alpha,\beta)$ has known summaries, so we get the answers in closed form — no grid, no sampling:
$$ \text{mean} = \frac{\alpha}{\alpha+\beta} = \frac{7}{11} \approx 0.636, \qquad \text{mode} = \frac{\alpha-1}{\alpha+\beta-2} = \frac{6}{9} \approx 0.667. $$

Those are *exactly* the numbers the grid produced (0.636 and 0.667) — the grid was approximating this closed form all along. Let's confirm and extract a credible interval:

```python
from scipy.stats import beta

alpha_post, beta_post = 1 + w, 1 + (n - w)        # Beta(7, 4)
print(f"posterior mean ≈ {beta.mean(alpha_post, beta_post):.3f}")     # 0.636
print(f"posterior mode ≈ {(alpha_post - 1)/(alpha_post + beta_post - 2):.3f}")  # 0.667

# 95% credible interval: the central interval holding 95% of posterior mass
lo, hi = beta.ppf([0.025, 0.975], alpha_post, beta_post)
print(f"95% credible interval: [{lo:.3f}, {hi:.3f}]")   # ≈ [0.348, 0.878]

# A direct probability statement you could never make with a frequentist CI:
p_more_than_half = 1 - beta.cdf(0.5, alpha_post, beta_post)
print(f"P(theta > 0.5 | data) = {p_more_than_half:.3f}")   # ≈ 0.828
```

> 💡 **Intuition:** That last line — "given the data, there's an 82.8% probability that more than half the globe is water" — is a sentence about $\theta$ itself, a direct probability statement about the unknown. Hold onto it; in §8 we'll see that a frequentist confidence interval, however similar its numbers, is *forbidden* from making that statement, and understanding why is the crux of the whole Bayesian-vs-frequentist divide.

> 🔧 **In practice:** Conjugacy is gorgeous when you can get it, and it's the backbone of Chapter 04's analytic methods. But it only exists for a handful of likelihood/prior pairings (Beta-Binomial, Gamma-Poisson, Normal-Normal), and the moment your model has more than one or two parameters or a non-conjugate prior — which is *almost always* in real work — the closed form evaporates and the evidence integral becomes intractable. That is precisely the wall that **Chapter 05 (MCMC: from Metropolis to NUTS)** is built to climb. Grid approximation taught you the concept; conjugacy gave you a clean special case; MCMC is the general-purpose engine. Keep all three in view.

---

## 6. The generative modeling mindset

Now I want to rewire how you *think* about a model, because this shift is what separates people who fit Bayesian models from people who *design* them. Here is the reframe:

> **A Bayesian model is a recipe for simulating data.**

That's it. The pair (prior, likelihood) is not just two ingredients of a formula — together they define a **joint distribution** $p(y, \theta) = p(y \mid \theta)\,p(\theta)$, and a joint distribution is a *machine you can run forward to produce fake datasets*. The recipe:

```
1. draw a parameter from the prior:     theta*  ~  p(theta)
2. draw data given that parameter:      y_sim   ~  p(y | theta*)
3. you now hold a synthetic dataset y_sim that this model considers plausible.
```

Run that loop many times and you get an ensemble of fake datasets — the kinds of data your model believes are possible *before it has seen anything real*. This forward simulation has a name: the **prior predictive distribution**,
$$ p(\tilde y) = \int p(\tilde y \mid \theta)\, p(\theta)\, d\theta. $$
(Notice it's the evidence integral from §3, but for *hypothetical* future data $\tilde y$ — the prior, pushed forward through the likelihood.) Inference — conditioning on the *actual* observed $y$ — runs the same machine the other direction: instead of generating $y$ from $\theta$, we pin $y$ to what we saw and ask which $\theta$'s could plausibly have produced it. **Forward to imagine, backward to learn.**

> 💡 **Intuition (McElreath's framing):** The design question is always *"how could this data have arisen?"* Write the simulator that answers it, and the model usually falls out of the simulator almost for free. If you can write a NumPy function that, given some parameters, spits out a fake dataset resembling yours, you have essentially written your likelihood — turning that simulator into a PyMC model is mostly transcription. Generative-first thinking is the most reliable model-building heuristic I know.

Why does this matter so much in practice? Because the forward direction is **checkable before you spend a second on inference**. You can simulate from the prior and *look* at the implied datasets:

```python
# Prior predictive for the globe model, by hand — no PyMC needed yet.
rng = np.random.default_rng(RANDOM_SEED)
n = 9
theta_draws = rng.beta(1, 1, size=10_000)          # draw theta from the Beta(1,1) prior
w_sim = rng.binomial(n, theta_draws)               # draw fake water-counts given each theta
# w_sim is now an ensemble of plausible "number of waters in 9 tosses" BEFORE seeing data.
print(np.bincount(w_sim) / len(w_sim))             # implied distribution over w = 0..9
```

If those simulated water-counts looked physically absurd — say the model insisted you'd almost never see 4 or 5 waters in 9 tosses — that would tell you your prior is broken *before* you ever touched the real data. This is a **prior predictive check**, and it is the first checkpoint of the Bayesian workflow. After fitting, you run the same idea on the *posterior* — simulate $\tilde y \sim p(\tilde y \mid y)$ and compare to the real $y$ — which is a **posterior predictive check** (Betancourt calls a passing check "retrodictive": the model retrodicts the data you already hold). Both checks, and the deeper validation tools (fake-data parameter recovery, simulation-based calibration), are nothing but disciplined uses of the model's forward simulator. We will build all of them properly in **Chapter 08 (The Bayesian Workflow)**; for now, just internalize that the generative view is what makes a model *falsifiable* instead of merely fitted.

> 🔧 **In practice (PyMC names the forward and backward directions explicitly):** In PyMC v5 the forward simulation is `pm.sample_prior_predictive(draws=500, random_seed=RANDOM_SEED)` (note the argument is `draws=`, not `samples=`), which fills `idata.prior` and `idata.prior_predictive`. The backward direction — conditioning on data — is `pm.sample(...)`, returning an `az.InferenceData` object with the posterior in `idata.posterior`. And forward-from-the-posterior is `pm.sample_posterior_predictive(idata, extend_inferencedata=True)`. Same model object, three uses of one generative machine. You'll meet PyMC's prior/posterior predictive machinery starting in **Chapter 03** (where prior predictive checks become a working discipline), and write full end-to-end PyMC models from **Chapter 04** onward.

---

## 7. Likelihood vs probability: a function of *what*?

I promised we'd hammer this, because it is the conceptual hinge that, once it clicks, makes half of statistics suddenly legible. The likelihood and "the probability of the data" are the *same algebraic expression* — and yet they are different objects, because they hold different things fixed.

Take the Binomial expression from the globe toss:
$$ f(w, \theta) = \binom{9}{w}\,\theta^{w}(1-\theta)^{9-w}. $$

**Read it as a probability** — fix the parameter $\theta$, let the data $w$ vary:
$$ p(w \mid \theta = 0.6) = \binom{9}{w}\,(0.6)^{w}(0.4)^{9-w}, \qquad w \in \{0,1,\dots,9\}. $$
This is a **probability distribution over the data**. Sum it over all possible $w$ and you get exactly 1 — there are ten possible outcomes and their probabilities partition certainty. It answers: *"if the bias really were 0.6, how often would I see each possible water-count?"*

**Read it as a likelihood** — fix the data $w = 6$, let the parameter $\theta$ vary:
$$ \mathcal{L}(\theta) = p(w = 6 \mid \theta) = \binom{9}{6}\,\theta^{6}(1-\theta)^{3}, \qquad \theta \in [0,1]. $$
This is a **function of the parameter $\theta$**. It is *not* a probability distribution over $\theta$. Integrate it over $\theta$ from 0 to 1 and you get $\binom{9}{6} B(7,4) = \binom{9}{6}\cdot\frac{6!\,3!}{10!} = 84 \cdot \frac{1}{840} = 0.1$ — **not 1.** It answers a comparative question: *"given that I saw 6 waters, which parameter values made that outcome more vs less probable?"* It ranks parameter values; it does not assign them probabilities.

```
                       hold θ fixed, vary the data  →  PROBABILITY (sums to 1 over data)
  f(w, θ) = (9 choose w) θ^w (1-θ)^(9-w)
                       hold data fixed, vary θ      →  LIKELIHOOD (does NOT integrate to 1 over θ)
```

> ⚠️ **Pitfall:** "The likelihood of $\theta = 0.6$ is 0.25" does **not** mean "there's a 25% probability that $\theta = 0.6$." Likelihood values are not probabilities of parameters; only their *relative* sizes carry meaning (the likelihood can even be rescaled by any positive constant without changing inference — which is why we could drop $\binom{9}{6}$ in the grid code). The thing that *is* a probability distribution over $\theta$ is the **posterior**, and the only way to turn a likelihood into one is to multiply by a prior and normalize — i.e., to apply Bayes' theorem. **Maximum likelihood** finds the $\theta$ that maximizes $\mathcal{L}$; it is the peak of the likelihood, and it equals the posterior mode *only when the prior is flat*. The frequentist stops at that peak and bolts on a standard error; the Bayesian carries the whole distribution forward.

> 💡 **Intuition (the deepest reason this distinction matters):** Because the likelihood is a function of the data given a parameter, *a likelihood by itself can never tell you the probability of a hypothesis* — it can only tell you the probability of data under a hypothesis. Going from "probability of data given hypothesis" to "probability of hypothesis given data" requires **inverting the conditional**, and inverting a conditional *requires a prior* (that's literally what the product rule says: $p(\theta\mid y)$ needs $p(\theta)$). There is no prior-free way to get a probability of a hypothesis. People who think they've done it have smuggled a prior in unnoticed — usually a flat one — and a flat prior is a *choice*, not the absence of one. This is the technical heart of why "let the data speak for itself" is a comforting fiction.

---

## 8. Bayesian vs frequentist, done honestly

You came in fluent in frequentist tools — point estimates, standard errors, p-values, confidence intervals — even if no one ever called them that. So let me draw the line cleanly, without tribalism. There are good statisticians on both sides; the goal here is not to win an argument but to know *exactly* what each framework lets you say, so you never accidentally claim more than your method delivers.

### What is random?

This is the fork in the road, and everything else follows from it.

- **Frequentist:** the parameter $\theta$ is a **fixed unknown constant** — it has one true value, full stop, and that value is not a random variable. What's random is the **data**, conceived as one draw from an infinite hypothetical population of repeated experiments. Any statistic you compute from the data (an estimate, an interval) is therefore also random — it would come out differently if you re-ran the study. Inference reasons about the long-run behavior of these procedures over imagined repetitions.
- **Bayesian:** the data, once observed, are **fixed** — you saw what you saw. What carries a probability distribution is **$\theta$**, not because $\theta$ physically fluctuates, but because *your knowledge of it is uncertain*. Inference reasons about your degree of belief, conditioned on the one dataset you actually have.

> 💡 **Intuition:** Frequentist inference holds $\theta$ fixed and imagines the data wobbling; Bayesian inference holds the data fixed and lets belief about $\theta$ spread out. One conditions on hypotheses and integrates over data; the other conditions on data and integrates over hypotheses. That single transposition of "what varies" generates almost every downstream difference.

### Confidence interval vs credible interval — the difference that bites

These two are constantly confused, and the confusion is not harmless. Recall our globe posterior gave a 95% **credible** interval of roughly $[0.348, 0.878]$.

**A 95% credible interval** is a statement about $\theta$ given *your* data and prior:
$$ P(\theta \in [0.348, 0.878] \mid y) = 0.95. $$
"Given everything I know, there is a 95% probability the true water-proportion lies in this interval." This is a direct probability statement about the parameter — and it is exactly the sentence people *want* to say.

**A 95% confidence interval** is a statement about the *procedure*, not about $\theta$:

> If I were to repeat this experiment indefinitely and compute an interval each time by this method, 95% of those intervals would contain the true $\theta$.

It is a property of the *method's long-run coverage*, not of the *one interval you're holding*. For your particular interval, $\theta$ is a fixed constant — it is either in there or it isn't — so the "probability it contains $\theta$" is, from the frequentist stance, either 0 or 1, just unknown which. The "95%" describes the factory that produced the interval, not this specific item off the line.

> ⚠️ **Pitfall (the one everybody commits):** Saying "there's a 95% probability $\theta$ is in *this* confidence interval" is **a misinterpretation of the frequentist CI** — yet it's what almost everyone, including textbooks and reviewers, says. The statement is *correct only for a credible interval*. So one of two things is true: either you computed a credible interval (Bayesian) and the sentence is licensed, or you computed a confidence interval (frequentist) and the sentence is wrong. The reason the confusion is so durable is that people reach for frequentist tools precisely *because* they want the Bayesian conclusion — and then misread the output as if it provided it. Knowing which object you're holding is not pedantry; it's the difference between a true and a false statement.

| | Confidence interval (frequentist) | Credible interval (Bayesian) |
|---|---|---|
| What's random | the interval (data varies over repetitions) | $\theta$ (uncertain given fixed data) |
| Probability claim | about the *procedure's* long-run coverage | directly about $\theta$: $P(\theta\in[\ell,u]\mid y)$ |
| "95% chance $\theta$ is inside" | **false** for a given interval | **true** |
| Needs a prior | no | yes |
| Uses data via | a chosen estimator + its sampling distribution | the likelihood (and prior) |

### p-values, and the likelihood principle

A **p-value** is $p(\text{data as or more extreme} \mid H_0)$ — the probability, *if the null hypothesis were true*, of observing data at least as extreme as what you got. Note what it is and isn't: it is a tail probability of the *data* under a fixed hypothesis (a frequentist, sampling-distribution object). It is **not** $p(H_0 \mid \text{data})$ — that would require Bayes and a prior. Confusing the two is the same conditional-inversion error as §7, now in hypothesis-testing clothes.

There's a sharper, more philosophical difference hiding here, and it's worth meeting once. The **likelihood principle** states that all the evidence in the data about $\theta$ is contained in the likelihood function. Bayesian inference obeys it automatically — the posterior depends on the data *only* through $p(y\mid\theta)$. Frequentist tail-probabilities **violate** it, because "data as or more extreme" depends on outcomes you *didn't* observe and on the experimenter's **stopping rule** (did you plan to flip 9 times, or to flip until you got 3 lands?). Two experiments with identical observed data but different stopping intentions yield *identical* Bayesian posteriors but *different* p-values. Whether that's a feature or a bug is a genuine, century-old debate — but it is a real philosophical difference, not mere bookkeeping, and you should know it exists.

### When they agree, and when the difference bites

I want to be scrupulously fair: **with a lot of data and a flat-ish prior, the two frameworks usually give numerically similar answers.** This is not a coincidence — it's the **Bernstein–von Mises theorem**: as $n \to \infty$, the posterior concentrates and becomes approximately Normal centered at the maximum-likelihood estimate, so a 95% credible interval and a 95% confidence interval nearly coincide. In the big-data, weak-prior regime, the choice of framework is often a choice of *vocabulary*, and a competent frequentist analysis and a competent Bayesian analysis land in the same place. Don't let anyone tell you Bayes is magic that fixes a fundamentally bad study.

The difference **bites** in exactly the situations where it matters most:

- **Small samples / weak data.** When the data don't pin $\theta$ down, the prior genuinely contributes, and the regularization it provides keeps estimates sane where MLE overfits or diverges.
- **Hierarchical structure.** When you have many related groups (schools, hospitals, users), Bayesian **partial pooling** shares information across them and shrinks noisy small-group estimates toward the population — automatically, as a *consequence* of the model, not a bolt-on. This is the headline result of **Chapter 10 (Hierarchical Models)**, and it routinely beats both "estimate each group alone" and "ignore groups entirely."
- **Decision-making under uncertainty.** A full posterior lets you compute the *expected utility* of an action by integrating over everything you don't know — the natural input to a decision. A point estimate plus a standard error throws away the shape of your uncertainty.
- **Propagating uncertainty into predictions.** The posterior predictive carries parameter uncertainty all the way into forecasts; plug-in prediction (use $\hat\theta$, ignore its uncertainty) is systematically overconfident.
- **Nuisance parameters.** Bayes *integrates them out* (marginalization, the sum rule) rather than plugging in point estimates — cleaner and honest about the uncertainty they contribute.

> 🔧 **In practice:** Use this rule of thumb. *Big clean data, simple model, just need a number?* Either framework is fine; pick the one your audience reads fluently. *Small or messy data, hierarchical structure, need calibrated uncertainty for a real decision, or want to encode domain knowledge?* The Bayesian machinery pays for its compute cost many times over. And be honest about that cost — which is the next point.

### Why go Bayesian — and what it costs

The honest case *for*: (i) **coherent uncertainty quantification** — a full posterior, not a point plus a standard error you'll misread anyway; (ii) priors let you **use domain knowledge and regularize**, with shrinkage falling out for free; (iii) **generative thinking** yields checkable, falsifiable models; (iv) a **uniform treatment** of hard models — hierarchies, latent variables, missing data, measurement error — without inventing a bespoke estimator for each; (v) **honest propagation** of uncertainty into predictions and decisions.

The honest costs: (i) **computation** — MCMC is slower than a closed-form fit and can fail in ways you must learn to diagnose (the whole of Chapters 05 and 07); (ii) **prior elicitation** takes real thought and is a place to make mistakes; (iii) you are *obligated to check your model* — prior predictive, diagnostics, posterior predictive — and skipping the checks forfeits most of the benefit. Bayesian methods buy coherence and calibrated uncertainty at the price of compute and discipline. That trade is usually worth it for serious work, but it is a trade, and a good mentor tells you so rather than selling you a religion.

---

## 9. The small world and the large world

I'll close the conceptual core with McElreath's metaphor, because it's the idea that makes everything in §8 honest and motivates every "check your model" instruction in the rest of the course.

> 📜 **Citation/Origin:** Richard McElreath, _Statistical Rethinking_ (2nd ed., 2020), Chapter 1 ("The Golem of Prague") and Chapter 2 ("Small Worlds and Large Worlds"). The metaphor is load-bearing for this whole course.

**The golem.** A statistical model, McElreath says, is a *golem* — the clay automaton of Prague legend, immensely powerful and perfectly obedient and utterly without wisdom. It does *exactly* what its assumptions instruct, with no judgment about whether those instructions make sense in the world. Bayes' theorem is the most obedient golem of all: feed it a prior and a likelihood and it returns *the* logically correct posterior — correct *with respect to the assumptions you gave it.*

**The small world** is the self-contained logical universe *inside* the model — the complete list of possibilities the model entertains (every $\theta$, every $\tilde y$) and the probabilities it assigns them. **Within the small world, Bayesian updating is provably optimal.** Given the model's assumptions, no procedure extracts more information from the data than the posterior does. That optimality is real and it is exactly what Cox's theorem (§1) guarantees: consistent reasoning *inside a fixed set of assumptions* is uniquely the probability calculus.

**The large world** is messy reality, where the model is actually deployed — where your assumptions are approximations, your likelihood is wrong in the fourth decimal place (and maybe the first), and possibilities you never enumerated keep happening. And here is the crucial, humbling fact:

> **Optimality in the small world is no guarantee whatsoever in the large world.**

A model can be flawlessly, optimally Bayesian about a small world that does not match reality, and its beautiful credible intervals can be confidently, precisely wrong. The posterior is only as trustworthy as the small world is a faithful map of the large one.

> 💡 **Intuition:** This is *why model checking exists.* The math (Cox, Bayes) guarantees you reason correctly *inside* the model; nothing guarantees the model matches the world. So you must check the join between small and large world empirically — which is exactly what **prior predictive checks** (is my small world even plausible before data?), **posterior predictive checks** (does my fitted small world reproduce the real data's features?), and **model expansion** (grow the small world when it can't) are *for*. McElreath's metaphor and Betancourt's rigorous workflow are the same lesson in two voices: be optimal inside the model, and be relentlessly skeptical that the model is the world. A Bayesian who reports a posterior without checking it has trusted a golem.

---

## 10. A fully worked recap: the base-rate problem end to end, in code

We computed the medical-test posterior by hand in §4; let me close the loop by treating it as a tiny *generative model* and verifying the hand arithmetic by simulation — exactly the forward-simulation discipline from §6, on a named, self-contained dataset (the diagnostic-test problem). This is the workflow spine in miniature: define the generative process → simulate → condition → check the analytic answer against the simulation.

```python
import numpy as np

RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)

# --- the generative model (a recipe to simulate one person's test result) ---
prevalence  = 0.001     # p(D)        : base rate
sensitivity = 0.99      # p(+ | D)    : true positive rate
fpr         = 0.01      # p(+ | ¬D)   : false positive rate  (1 - specificity)

N = 2_000_000           # simulate a big population, forward through the model
has_disease  = rng.random(N) < prevalence                       # step 1: draw θ (sick?) from prior
p_pos        = np.where(has_disease, sensitivity, fpr)          # step 2: likelihood of a +
tests_pos    = rng.random(N) < p_pos                            # step 2: draw the data (test result)

# --- condition on the data: among those who tested +, what fraction are sick? ---
ppv_sim = has_disease[tests_pos].mean()        # P(D | +), estimated by simulation
print(f"simulated  P(D | +) ≈ {ppv_sim:.4f}")  # ≈ 0.090

# --- the exact analytic posterior from §4, for comparison ---
num = sensitivity * prevalence
den = sensitivity * prevalence + fpr * (1 - prevalence)
print(f"analytic   P(D | +)  = {num/den:.4f}")  # 0.0902
```

> 🩺 **Diagnostic (what to look for):** The simulated and analytic values agree to about two decimals (the simulation wobbles by Monte Carlo noise that shrinks as `N` grows). That agreement is a **fake-data sanity check** — the cheapest, most powerful habit in this course: whenever you can compute something two ways (here, by hand and by simulation), do, and confirm they match. When we move to PyMC models where the analytic answer doesn't exist, this *simulate-and-recover* discipline becomes how we trust our code at all. It is the seed of fake-data parameter recovery and simulation-based calibration in Chapter 08. Notice, too, that the simulation makes the base-rate effect tangible: of the roughly 22,000 positives in two million people, only ~2,000 are genuinely sick — you can literally count the false-positive flood.

> 🗺️ **A note on the datasets.** The globe toss and the diagnostic-test problem are deliberately the smallest *complete* Bayesian analyses there are — the canonical pedagogical toy problems (McElreath opens *Statistical Rethinking* with the globe for exactly this reason), and synthetic-with-known-truth examples like these are the right first companions because you control the answer and can check the machinery against it. From here the course graduates to the recurring real datasets you'll see again and again: the **Howell !Kung** census (`Howell1.csv`) as the flagship for linear regression starting in **Chapter 09**, **Eight Schools** as the canonical hierarchical/partial-pooling teaching set in **Chapter 10**, and **Radon** and the **Palmer Penguins** for multilevel and GLM work thereafter. Same engine you just ran by hand — bigger, messier worlds for it to run in.

---

## ⚠️ Common errors & how to fix them

These are foundations-level conceptual traps — the ones that quietly corrupt analyses for months. (Installation and PyMC-API errors live in **Chapter 00**; modeling/sampling errors arrive in Chapters 05 and 07.)

| Symptom | Cause | Fix |
|---|---|---|
| "The 95% CI means there's a 95% chance $\theta$ is inside" — stated for a **frequentist** interval | Conflating a confidence interval (a property of the *procedure*) with a credible interval (a probability about $\theta$) | Only the **credible** interval licenses that sentence. Either compute a Bayesian credible interval, or downgrade the claim to "the procedure covers 95% of the time." See §8 table. |
| Treating the likelihood as a probability distribution over $\theta$ ("there's a 0.25 probability $\theta=0.6$ because the likelihood is 0.25") | Forgetting the likelihood is a function of $\theta$ that does **not** integrate to 1 over $\theta$ | To get a probability over $\theta$ you must multiply by a prior and normalize — i.e. compute the posterior (§7). Maximum likelihood = posterior mode *only* under a flat prior. |
| "I used a flat prior so my analysis is objective / assumption-free" | A flat prior is a *choice*, not the absence of one — and it's not even flat after a nonlinear reparameterization | Own the prior explicitly; check it with a **prior predictive** simulation (§6). "No prior" is impossible; "unexamined prior" is the real risk. |
| Answering the base-rate problem with the test's accuracy (~99%) instead of its positive predictive value (~9%) | **Base-rate neglect** — ignoring the prior odds when the event is rare | Always combine the likelihood ratio with the **prior odds** (§4). Compute PPV, not accuracy, for rare-event classifiers. |
| Prior predictive draws look physically absurd (e.g. heights of $10^6$ cm, water-counts that can't occur) | Vague priors on unstandardized quantities create a nonsensical small world | Standardize predictors and use weakly-informative priors (Chapter 03); **re-run the prior predictive** until the implied data are plausible. |
| Reporting a posterior without ever checking the model | Trusting small-world optimality as if it guaranteed large-world correctness | Run prior predictive → diagnose the fit (R-hat, ESS, divergences — defined in Chapter 07) → posterior predictive (the spine, Chapter 08). An unchecked golem is not evidence. |
| Grid approximation gives a jagged or truncated posterior | Grid too coarse, or its range clips real posterior mass | Use a finer grid and verify the range covers where the posterior has support; remember grids collapse beyond ~2–3 parameters (use MCMC, Chapter 05). |
| p-value read as $p(H_0 \mid \text{data})$ | Inverting the conditional without a prior | A p-value is $p(\text{data this extreme}\mid H_0)$, a statement about data, not about $H_0$. Only Bayes gives $p(H_0\mid\text{data})$, and only with a prior (§8). |

---

## 🧪 Exercises

Work these with pen, paper, and a Python REPL. Hints follow each.

**1. (Conceptual) The two interpretations.** For each statement, decide whether it only makes sense under the frequency interpretation, only under the degree-of-belief interpretation, or both: (a) "This coin lands heads with probability 0.5." (b) "There's a 60% probability it rained in Lagos yesterday." (c) "The probability that this specific 95% confidence interval contains $\theta$ is 0.95." (d) "Given the data, there's an 83% probability the globe is more than half water."
> _Hint:_ Ask whether the event is repeatable, and whether the claim is about a procedure or about a one-time state of the world. One of these is a *false* statement under its natural framework — find it.

**2. (By hand) Re-test the patient.** In the base-rate problem (§4), the patient tested positive once and the posterior was $p(D\mid +)\approx 0.09$. They take an *independent* second test, which also comes back positive. Compute the new posterior. *Yesterday's posterior is today's prior:* use 0.09 as the new prior for $p(D)$ and apply Bayes again. Then comment on why two positives are so much more convincing than one.
> _Hint:_ Use the odds form. Prior odds $\approx 0.09/0.91$; multiply by the likelihood ratio $99$ again. You should land near $p(D\mid ++)\approx 0.91$. Notice that the likelihood ratio is the same both times, but the *prior odds* are no longer lopsided.

**3. (Coding) Grid vs conjugacy must agree.** Extend the §5 globe code: write a function `posterior_grid(w, n, grid_size)` returning the normalized grid posterior under a flat prior, and compare its mean and a 95% interval against the exact $\mathrm{Beta}(1+w,\,1+n-w)$. Verify they converge as `grid_size` grows from 20 → 50 → 500. Then change the prior in the grid to a `Beta(2, 2)` (mild belief that $\theta$ is near 0.5) and confirm the grid still matches the analytic $\mathrm{Beta}(2+w,\,2+n-w)$.
> _Hint:_ Evaluate the Beta(2,2) prior on the grid with `scipy.stats.beta.pdf(theta_grid, 2, 2)`. The grid method doesn't care that the prior changed — that generality is the whole point.

**4. (Coding + conceptual) Sequential updating is order-free.** Take the 9 globe tosses and feed them to the §5 sequential-update loop in three different random orders (use `np.random.default_rng(8927).permutation`). Plot all three final posteriors on one axis. Explain in one sentence, referencing the product rule, why they are identical.
> _Hint:_ Updating multiplies likelihood factors; multiplication commutes. The final posterior depends on the data only through the counts $(w, n)$ — a concrete instance of the likelihood principle.

**5. (Conceptual) Likelihood is not a probability over $\theta$.** For the globe data ($w=6$, $n=9$), the likelihood is $\mathcal L(\theta)=\binom{9}{6}\theta^6(1-\theta)^3$. (a) At what $\theta$ is it maximized? (b) Compute $\int_0^1 \mathcal L(\theta)\,d\theta$ and confirm it is **not** 1. (c) What prior turns this likelihood into a *proper* posterior, and what is that posterior's name?
> _Hint:_ (a) maximize $6\ln\theta + 3\ln(1-\theta)$ → $\theta=2/3$. (b) the integral is $\binom{9}{6}B(7,4)=0.1$. (c) a flat $\mathrm{Beta}(1,1)$ prior; the posterior is $\mathrm{Beta}(7,4)$.

**6. (Decision-making, stretch) From posterior to action.** You'll bet on whether the globe is more than half water. A correct "yes" wins \$1; an incorrect "yes" loses \$3; saying "no" pays \$0. Using the §5 posterior $\mathrm{Beta}(7,4)$, compute $P(\theta>0.5\mid y)$, then the *expected* payoff of betting "yes," and decide. This is a full posterior earning its keep — a point estimate alone cannot answer it.
> _Hint:_ $P(\theta>0.5\mid y)\approx 0.828$ (from §5). Expected payoff of "yes" $= 0.828(+1) + 0.172(-3)$. Is it positive? Bayesian decision theory is just *maximize expected utility under the posterior* — a thread we pick up in Chapter 11.

---

## 📚 Resources & further reading

The philosophical and conceptual anchors for this chapter, each with a specific pointer.

- **E. T. Jaynes, _Probability Theory: The Logic of Science_ (2003)** — Chapters 1–2 derive the sum and product rules from Cox's desiderata. This is *the* source for "probability as extended logic" and the deepest answer to "why Bayesian." Dense but transformative.
- **Richard McElreath, _Statistical Rethinking_, 2nd ed. (2020)** — Chapter 1 ("The Golem of Prague") for the golem metaphor; Chapter 2 ("Small Worlds and Large Worlds") for the globe-toss, grid approximation, and the small/large-world distinction used throughout this chapter. Book page: <https://xcelab.net/rm/>. Free 2023 lectures and code: <https://github.com/rmcelreath/stat_rethinking_2023>.
- **PyMC port of _Statistical Rethinking_** — runnable PyMC v5 versions of McElreath's models, including the globe toss and grid/quadratic approximations: <https://github.com/pymc-devs/pymc-resources>. Mine this when you want the PyMC-native version of anything here.
- **Martin, Kumar, Lao, _Bayesian Modeling and Computation in Python_ (2021)** — free online at <https://bayesiancomputationbook.com>. Chapter 1 ("Bayesian Inference") parallels this chapter's Bayes-theorem decomposition, grid approximation, and conjugacy in idiomatic PyMC/ArviZ — the direct companion to this course.
- **Osvaldo Martin, _Bayesian Analysis with Python_, 3rd ed. (2024)** — a gentle, code-first tour of exactly these foundations with PyMC.
- **Gelman, Carlin, Stern, Dunson, Vehtari, Rubin, _Bayesian Data Analysis_ (BDA3, 2013)** — the reference. Chapter 1 for probability and Bayes' theorem; free PDF: <http://www.stat.columbia.edu/~gelman/book/>.
- **Gelman, Hill, Vehtari, _Regression and Other Stories_ (2020)** — free PDF and examples at <https://avehtari.github.io/ROS-Examples/>. Chapter 9 ("Prediction and Bayesian inference") has an unusually clear, honest treatment of the confidence-vs-credible-interval distinction in §8.
- **Gelman et al., "Bayesian Workflow" (2020)** — <https://arxiv.org/abs/2011.01808>. The forward pointer for §6 and §9: how prior/posterior predictive checks and model expansion bridge small world and large world. We unpack it fully in Chapter 08.
- **Betancourt, "Towards A Principled Bayesian Workflow"** — <https://betanalpha.github.io/assets/case_studies/principled_bayesian_workflow.html>. Source of the "retrodictive check" terminology and the rigorous generative-checking discipline previewed here.
- **Wikipedia, "Credible interval"** — <https://en.wikipedia.org/wiki/Credible_interval>. Concise, correct cross-check of the fixed-vs-random framing from §8 if you want a second phrasing.
- **PyMC "Prior and Posterior Predictive Checks" core notebook** — <https://www.pymc.io/projects/docs/en/latest/learn/core_notebooks/posterior_predictive.html>. The verified v5 idioms (`sample_prior_predictive(draws=...)`, `sample_posterior_predictive(..., extend_inferencedata=True)`) behind §6's forward-simulation note.

---

## ➡️ What's next

You now own the engine: probability as the unique logic of uncertainty, Bayes' theorem as the product rule rearranged, the four named pieces, the generative mindset, and the honest line between Bayesian and frequentist reasoning. The natural next question is the one this chapter kept deferring — *where does the prior come from, and how do I choose it well?* That's **Chapter 02 — Priors I: Concepts and Common Priors**, where we meet flat vs proper, conjugate, weakly-informative, regularizing, and PC priors, and build the catalog of distributions with a rule for exactly when to reach for each. After that, **Chapter 03 (Priors II)** turns the prior predictive checks we previewed in §6 into a working discipline. And once priors are solid, **Chapter 04** generalizes the grid and conjugacy methods from §5 into the full analytic/grid/quadratic toolkit — before **Chapter 05** finally unleashes MCMC to do what no closed form can. The whole arc — generative model → prior predictive check → fit → diagnose → posterior predictive check → expand — is the **Bayesian workflow** of **Chapter 08**, and every chapter between here and there is teaching you one of its moves. Onward.
